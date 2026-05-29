"""AnchorMoE-CR model.

Core components:

    MVRE: Multi-View Representation Embedding
    DPAR: Diversified Posterior-Anchor Routing
    RGC : Reliability-Gated Composition
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Patch-grid helpers
# ---------------------------------------------------------------------------


def _build_patch_meta(seq_len: int, patch_len: int, stride: int, device=None) -> Dict:
    patch_len = int(patch_len)
    stride = int(stride)
    if patch_len <= 0 or stride <= 0:
        raise ValueError("patch_len and stride must be positive")

    pad_left = max(0, patch_len - stride)
    padded_length = int(seq_len) + pad_left
    if padded_length < patch_len:
        raise ValueError(
            f"Invalid patch grid: padded_length={padded_length} < patch_len={patch_len}"
        )

    num_patches = ((padded_length - patch_len) // stride) + 1
    idx = torch.arange(num_patches, device=device, dtype=torch.long)
    padded_start = idx * stride
    patch_start = padded_start - pad_left
    patch_end = patch_start + patch_len
    effective_start = patch_start.clamp(min=0, max=seq_len)
    effective_end = patch_end.clamp(min=0, max=seq_len)

    return {
        "seq_len": int(seq_len),
        "patch_len": patch_len,
        "stride": stride,
        "pad_left": pad_left,
        "padded_length": padded_length,
        "num_patches": int(num_patches),
        "effective_start": effective_start,
        "effective_end": effective_end,
    }


def _patch_padding_mask(x_mark_enc, patch_meta: Dict, device) -> Optional[torch.Tensor]:
    """Convert a per-timestep mask to a per-patch validity mask."""
    if x_mark_enc is None:
        return None
    pm = x_mark_enc.bool() if x_mark_enc.dtype == torch.bool else (x_mark_enc > 0.5)
    lengths = pm.sum(dim=1).to(torch.long)
    eff_start = patch_meta["effective_start"].to(device=device)
    eff_end = patch_meta["effective_end"].to(device=device)
    valid = (eff_end.unsqueeze(0) > eff_start.unsqueeze(0)) & (
        eff_start.unsqueeze(0) < lengths.unsqueeze(1)
    )
    return valid


def _unfold_with_patch_meta(x: torch.Tensor, patch_meta: Dict) -> torch.Tensor:
    """Unfold x from (B,T,C) to aligned windows (B,C,P,L)."""
    if x.dim() != 3:
        raise ValueError("x must have shape (B,T,C)")
    pad_left = int(patch_meta["pad_left"])
    patch_len = int(patch_meta["patch_len"])
    stride = int(patch_meta["stride"])
    expected = int(patch_meta["num_patches"])
    x_pad = F.pad(x, (0, 0, pad_left, 0))
    windows = x_pad.permute(0, 2, 1).unfold(dimension=2, size=patch_len, step=stride)
    if int(windows.size(2)) != expected:
        raise AssertionError(
            f"Patch count mismatch in spectral unfold: {int(windows.size(2))} vs {expected}"
        )
    return windows


# ---------------------------------------------------------------------------
# Component 1: MVRE (Multi-View Representation Embedding)
# ---------------------------------------------------------------------------


class _SinCosPosEmb(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[: x.size(1)].unsqueeze(0)


class PatchEmbedding(nn.Module):
    """Channel-joint Conv1d patch embedding.

    The output token is the temporal input to MVRE.
    """

    def __init__(self, c_in: int, d_model: int, patch_len: int, stride: int, dropout: float = 0.1):
        super().__init__()
        self.patch_len = int(patch_len)
        self.stride = int(stride)
        self.conv = nn.Conv1d(c_in, d_model, kernel_size=self.patch_len, stride=self.stride)
        self.norm = nn.LayerNorm(d_model)
        self.pos = _SinCosPosEmb(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        meta = _build_patch_meta(
            seq_len=int(x.size(1)), patch_len=self.patch_len, stride=self.stride, device=x.device
        )
        x_pad = F.pad(x.permute(0, 2, 1), (int(meta["pad_left"]), 0))
        tokens = self.conv(x_pad).transpose(1, 2)
        if int(tokens.size(1)) != int(meta["num_patches"]):
            raise AssertionError(f"patch count mismatch: {int(tokens.size(1))} vs {int(meta['num_patches'])}")
        tokens = self.drop(self.pos(self.norm(tokens)))
        return tokens, meta


class SpectralBandEncoder(nn.Module):
    """Aligned FFT-band log-energy encoder."""

    def __init__(self, c_in: int, d_model: int, num_bands: int = 4, dropout: float = 0.1):
        super().__init__()
        self.c_in = int(c_in)
        self.num_bands = max(1, int(num_bands))
        hidden = max(16, d_model // 2)
        self.proj = nn.Sequential(
            nn.LayerNorm(self.c_in * self.num_bands),
            nn.Linear(self.c_in * self.num_bands, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(self, x_raw: torch.Tensor, patch_meta: Dict, eps: float = 1e-8) -> torch.Tensor:
        windows = _unfold_with_patch_meta(x_raw, patch_meta)  # (B,C,P,L)
        fft_out = torch.fft.rfft(windows, dim=-1)
        power = fft_out.real.pow(2) + fft_out.imag.pow(2)  # (B,C,P,F)
        freq_dim = int(power.size(-1))

        # Partition the RFFT bins into approximately equal bands. Empty bands
        # are avoided by clamping end > start.
        band_feats = []
        for b in range(self.num_bands):
            start = int(math.floor(b * freq_dim / self.num_bands))
            end = int(math.floor((b + 1) * freq_dim / self.num_bands))
            end = max(start + 1, min(end, freq_dim))
            band_energy = power[..., start:end].sum(dim=-1)
            band_feats.append(torch.log1p(band_energy.clamp_min(eps)))
        stats = torch.stack(band_feats, dim=-1)  # (B,C,P,Bands)
        stats = stats.permute(0, 2, 1, 3).flatten(start_dim=2)  # (B,P,C*Bands)
        return self.proj(stats)


class MVRE(nn.Module):
    """Multi-View Representation Embedding for local evidence construction.

    MVRE fuses temporal, spectral, and contextual views into one local
    evidence token. The same token is used by DPAR, RGC, and attribution.
    """

    def __init__(
        self,
        c_in: int,
        d_model: int,
        num_bands: int = 4,
        dropout: float = 0.1,
        use_mvre: bool = True,
        spectral_gate_init: float = 0.0,
        relevance_gate_init: float = -2.0,
    ):
        super().__init__()
        self.use_mvre = bool(use_mvre)
        self.time_refine = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.time_refine_gate_logit = nn.Parameter(torch.tensor(-2.0))
        self.freq_proj = SpectralBandEncoder(c_in, d_model, num_bands=num_bands, dropout=dropout)
        self.relevance_proj = nn.Sequential(
            nn.Linear(1, d_model),
            nn.GELU(),
        )
        self.evidence_norm = nn.LayerNorm(d_model)
        self.evidence_drop = nn.Dropout(dropout)
        self.query_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )
        self.spectral_gate_logit = nn.Parameter(torch.tensor(float(spectral_gate_init)))
        self.relevance_gate_logit = nn.Parameter(torch.tensor(float(relevance_gate_init)))

    def forward(self, tokens: torch.Tensor, x_raw: torch.Tensor, patch_meta: Dict,
                mask: Optional[torch.Tensor] = None, eps: float = 1e-6):
        if mask is not None:
            mf = mask.unsqueeze(-1).float()
            global_repr = (tokens * mf).sum(dim=1, keepdim=True) / mf.sum(dim=1, keepdim=True).clamp_min(eps)
        else:
            global_repr = tokens.mean(dim=1, keepdim=True)

        relevance = (tokens * global_repr).sum(dim=-1, keepdim=True) / math.sqrt(tokens.size(-1))

        if self.use_mvre:
            time_refine_gate = torch.sigmoid(self.time_refine_gate_logit)
            time_view = tokens + time_refine_gate * self.time_refine(tokens)
            freq_view = self.freq_proj(x_raw, patch_meta)
            spectral_gate = torch.sigmoid(self.spectral_gate_logit)
            relevance_view = self.relevance_proj(relevance)
            relevance_gate = torch.sigmoid(self.relevance_gate_logit)
            evidence_token = self.evidence_norm(
                time_view + spectral_gate * freq_view + relevance_gate * relevance_view
            )
            evidence_token = self.evidence_drop(evidence_token)
        else:
            time_view = tokens
            freq_view = torch.zeros_like(tokens)
            relevance_view = torch.zeros_like(tokens)
            spectral_gate = tokens.new_tensor(0.0)
            relevance_gate = tokens.new_tensor(0.0)
            time_refine_gate = tokens.new_tensor(0.0)
            evidence_token = tokens

        query = self.query_proj(evidence_token)
        aux = {
            "evidence_token": evidence_token,
            "mvre_relevance": relevance,
            "spectral_gate": spectral_gate.detach(),
            "relevance_gate": relevance_gate.detach(),
            "time_refine_gate": time_refine_gate.detach(),
        }
        return evidence_token, query, aux


# ---------------------------------------------------------------------------
# Component 2: DPAR (Diversified Posterior-Anchor Routing)
# ---------------------------------------------------------------------------


def _orthogonal_anchor_loss(dynamic_anchors: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Mean squared off-diagonal cosine similarity among dynamic anchors."""
    if dynamic_anchors.dim() != 3 or dynamic_anchors.size(1) <= 1:
        return dynamic_anchors.new_tensor(0.0)
    K = int(dynamic_anchors.size(1))
    an = F.normalize(dynamic_anchors, p=2, dim=-1, eps=eps)
    gram = torch.bmm(an, an.transpose(1, 2))
    eye = torch.eye(K, device=dynamic_anchors.device, dtype=dynamic_anchors.dtype).unsqueeze(0)
    off = ((gram - eye) * (1.0 - eye)).pow(2)
    return (off.sum(dim=(1, 2)) / float(K * (K - 1))).mean()


class DPARouter(nn.Module):
    """Diversified Posterior-Anchor Routing.

    DPAR assigns local evidence tokens to evidence types and constructs
    posterior anchors as routing-weighted evidence centroids. Orthogonality on
    posterior anchors discourages redundant evidence types.
    """

    def __init__(self, d_model: int, num_groups: int, router_topk: int = 2,
                 dropout: float = 0.1, use_dpar: bool = True,
                 router_temperature: float = 1.0):
        super().__init__()
        self.d_model = int(d_model)
        self.K = int(num_groups)
        self.topk = max(1, min(int(router_topk), self.K))
        self.use_dpar = bool(use_dpar)
        self.router_temperature = max(1e-4, float(router_temperature))
        self.router = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, self.K),
        )

    def forward(self, query: torch.Tensor, tokens: torch.Tensor,
                mask: Optional[torch.Tensor] = None, eps: float = 1e-6):
        logits = self.router(query)  # (B,P,K)
        train_logits = logits / self.router_temperature

        soft_probs = torch.softmax(train_logits, dim=-1)
        if self.topk < self.K:
            topk_vals, topk_idx = train_logits.topk(self.topk, dim=-1)
            topk_w = torch.softmax(topk_vals, dim=-1)
            token2group = torch.zeros_like(train_logits).scatter(-1, topk_idx, topk_w)
        else:
            topk_idx = torch.arange(self.K, device=query.device).view(1, 1, self.K).expand(query.size(0), query.size(1), -1)
            token2group = soft_probs

        if mask is not None:
            token2group = token2group * mask.unsqueeze(-1).float()
            soft_probs_masked = soft_probs * mask.unsqueeze(-1).float()
        else:
            soft_probs_masked = soft_probs

        routing_for_anchor = soft_probs_masked
        mass = routing_for_anchor.sum(dim=1).clamp_min(eps)  # (B,K)
        posterior_anchors = torch.einsum("bpk,bpd->bkd", routing_for_anchor, tokens) / mass.unsqueeze(-1)

        if self.use_dpar:
            anchor_div_loss = _orthogonal_anchor_loss(posterior_anchors)
        else:
            anchor_div_loss = tokens.new_tensor(0.0)

        valid = mask.float() if mask is not None else torch.ones(query.shape[:2], device=query.device, dtype=query.dtype)
        entropy = -(soft_probs.clamp_min(1e-9).log() * soft_probs).sum(dim=-1)
        routing_entropy = (entropy * valid).sum() / valid.sum().clamp_min(1.0)
        usage = token2group.sum(dim=(0, 1)) / token2group.sum().clamp_min(eps)

        aux = {
            "token2group": token2group,
            "routing_probs": token2group,
            "evidence_type": token2group.detach().argmax(dim=-1),
            "routing_logits": logits,
            "routing_soft_probs": soft_probs_masked,
            "routing_topk_idx": topk_idx,
            "posterior_anchors": posterior_anchors,
            "anchor_div_loss": anchor_div_loss,
            "routing_entropy": routing_entropy.detach(),
            "expert_usage": usage.detach(),
        }
        return token2group, aux


# ---------------------------------------------------------------------------
# Component 3: RGC (Reliability-Gated Composition)
# ---------------------------------------------------------------------------


class PatchLevelExpertWithConfidence(nn.Module):
    """Patch-level expert with a reliability branch."""

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d_model, d_model),
        )
        self.norm = nn.LayerNorm(d_model)
        self.conf_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, max(16, d_model // 2)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(max(16, d_model // 2), 1),
        )

    def forward(self, tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.norm(tokens + self.feature(tokens))
        conf_logit = self.conf_head(z).squeeze(-1)
        conf = torch.sigmoid(conf_logit)
        return z, conf, conf_logit


class RGC(nn.Module):
    """Reliability-Gated Composition.

    RGC computes expert evidence, estimates reliability, and composes final
    logits as an additive sum of patch-level class evidence.
    """

    def __init__(self, d_model: int, num_groups: int, num_class: int,
                 dropout: float = 0.1, use_confidence: bool = True,
                 binding_conf_power: float = 1.0,
                 composition_conf_power: float = 1.0):
        super().__init__()
        self.d_model = int(d_model)
        self.K = int(num_groups)
        self.num_class = int(num_class)
        self.use_confidence = bool(use_confidence)
        self.binding_conf_power = max(0.0, float(binding_conf_power))
        self.composition_conf_power = max(0.0, float(composition_conf_power))
        self.experts = nn.ModuleList([
            PatchLevelExpertWithConfidence(d_model, dropout=dropout)
            for _ in range(self.K)
        ])
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_class),
        )

    def forward(self, tokens: torch.Tensor, token2group: torch.Tensor,
                mask: Optional[torch.Tensor] = None, eps: float = 1e-6):
        features, confs, conf_logits = [], [], []
        for expert in self.experts:
            z, c, logit = expert(tokens)
            features.append(z)
            confs.append(c)
            conf_logits.append(logit)

        expert_features = torch.stack(features, dim=2)      # (B,P,K,D)
        expert_confidence = torch.stack(confs, dim=2)       # (B,P,K)
        confidence_logits = torch.stack(conf_logits, dim=2) # (B,P,K)

        if not self.use_confidence:
            effective_conf = torch.ones_like(expert_confidence)
        else:
            effective_conf = expert_confidence.clamp_min(eps).pow(self.binding_conf_power)

        expert_class_logits = self.classifier(expert_features)  # (B,P,K,C)
        weights = token2group * effective_conf
        patch_confidence = weights.sum(dim=-1).clamp_min(0.0)  # (B,P)
        if mask is not None:
            patch_confidence = patch_confidence * mask.float()

        importance_scores = patch_confidence.clamp_min(eps).pow(self.composition_conf_power)
        if mask is not None:
            importance_scores = importance_scores * mask.float()
        patch_importance = importance_scores / importance_scores.sum(dim=1, keepdim=True).clamp_min(eps)
        if mask is not None:
            patch_importance = patch_importance * mask.float()
            patch_importance = patch_importance / patch_importance.sum(dim=1, keepdim=True).clamp_min(eps)

        routing_only_logits = torch.einsum("bpk,bpkc->bpc", token2group, expert_class_logits)
        routing_mass = token2group.sum(dim=-1).clamp_min(0.0)
        if mask is not None:
            routing_mass = routing_mass * mask.float()
        confidence_free_importance = routing_mass / routing_mass.sum(dim=1, keepdim=True).clamp_min(eps)
        if mask is not None:
            confidence_free_importance = confidence_free_importance * mask.float()
            confidence_free_importance = confidence_free_importance / confidence_free_importance.sum(dim=1, keepdim=True).clamp_min(eps)
        confidence_free_patch_contribution = routing_only_logits * confidence_free_importance.unsqueeze(-1)

        patch_logits = torch.einsum("bpk,bpkc->bpc", weights, expert_class_logits)
        patch_contribution = patch_logits * patch_importance.unsqueeze(-1)
        out = patch_contribution.sum(dim=1)

        if self.use_confidence:
            attr_conf = expert_confidence
        else:
            attr_conf = torch.ones_like(expert_confidence)
        attr_scores = (token2group * attr_conf).sum(dim=-1).clamp_min(0.0)
        if mask is not None:
            attr_scores = attr_scores * mask.float()
        patch_importance_for_attr = attr_scores / attr_scores.sum(dim=1, keepdim=True).clamp_min(eps)
        if mask is not None:
            patch_importance_for_attr = patch_importance_for_attr * mask.float()
            patch_importance_for_attr = patch_importance_for_attr / patch_importance_for_attr.sum(dim=1, keepdim=True).clamp_min(eps)

        aux = {
            "expert_features": expert_features,
            "expert_confidence": expert_confidence,
            "confidence_logits": confidence_logits,
            "expert_class_logits": expert_class_logits,
            "patch_confidence": patch_confidence,
            "patch_importance": patch_importance,
            "patch_importance_for_attr": patch_importance_for_attr,
            "evidence_weight": patch_importance,
            "evidence_strength": patch_confidence,
            "evidence_logits_clean": routing_only_logits,
            "routing_only_logits": routing_only_logits,
            "confidence_free_patch_contribution": confidence_free_patch_contribution,
            "per_patch_logits": patch_logits,
            "patch_logits": patch_logits,
            "patch_contribution": patch_contribution,
        }
        return out, aux


# ---------------------------------------------------------------------------
# Top-level model
# ---------------------------------------------------------------------------


class Model(nn.Module):
    """AnchorMoE classifier with MVRE, DPAR, and RGC."""

    def __init__(self, args):
        super().__init__()
        self.task_name = args.task_name
        self.d_model = int(args.d_model)
        self.num_class = int(getattr(args, "num_class", 10))
        self.num_groups = int(getattr(args, "num_groups", 4))
        self.patch_len = int(getattr(args, "patch_len", 16))
        self.stride = int(getattr(args, "stride", max(1, self.patch_len // 2)))
        self.anchor_div_lambda = float(getattr(args, "anchor_div_lambda", 0.05))
        self.conf_lambda = float(getattr(args, "conf_lambda", 0.01))
        self.use_mvre = bool(getattr(args, "use_mvre", True))
        self.use_dpar = bool(getattr(args, "use_dpar", True))
        self.use_confidence = bool(getattr(args, "use_confidence", True))

        self.patch_embed = PatchEmbedding(
            c_in=int(args.enc_in),
            d_model=self.d_model,
            patch_len=self.patch_len,
            stride=self.stride,
            dropout=float(args.dropout),
        )
        self.mvre = MVRE(
            c_in=int(args.enc_in),
            d_model=self.d_model,
            num_bands=int(getattr(args, "mvre_num_freq_bands", 4)),
            dropout=float(args.dropout),
            use_mvre=self.use_mvre,
            spectral_gate_init=float(getattr(args, "mvre_spectral_gate_init", 0.0)),
            relevance_gate_init=float(getattr(args, "mvre_relevance_gate_init", -2.0)),
        )
        self.router = DPARouter(
            d_model=self.d_model,
            num_groups=self.num_groups,
            router_topk=int(getattr(args, "router_topk", min(2, self.num_groups))),
            dropout=float(args.dropout),
            use_dpar=self.use_dpar,
            router_temperature=float(getattr(args, "router_temperature", 1.0)),
        )
        self.rgc = RGC(
            d_model=self.d_model,
            num_groups=self.num_groups,
            num_class=self.num_class,
            dropout=float(args.dropout),
            use_confidence=self.use_confidence,
            binding_conf_power=float(getattr(args, "binding_conf_power", 1.0)),
            composition_conf_power=float(getattr(args, "composition_conf_power", 1.0)),
        )
        self.force_loss_aux = self.anchor_div_lambda > 0.0 or self.conf_lambda > 0.0

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        migrated = {}
        for key, value in state_dict.items():
            new_key = key
            if new_key.startswith("mvse."):
                new_key = "mvre." + new_key[len("mvse."):]
            if new_key.startswith("uadb."):
                new_key = "rgc." + new_key[len("uadb."):]
            migrated[new_key] = value
        return super().load_state_dict(migrated, strict=strict, assign=assign)

    @staticmethod
    def _normalize_importance(scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        scores = scores * mask.float()
        return scores / scores.sum(dim=1, keepdim=True).clamp_min(1e-6)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None,
                return_aux: bool = False):
        return_aux = bool(return_aux or (self.training and self.force_loss_aux))

        tokens, patch_meta = self.patch_embed(x_enc)
        patch_mask = _patch_padding_mask(x_mark_enc, patch_meta, x_enc.device)
        if patch_mask is None:
            patch_mask = torch.ones(
                x_enc.size(0), int(patch_meta["num_patches"]),
                device=x_enc.device, dtype=torch.bool,
            )

        evidence_token, query, mvre_aux = self.mvre(tokens, x_enc, patch_meta, patch_mask)
        token2group, router_aux = self.router(query, evidence_token, patch_mask)
        out, rgc_aux = self.rgc(evidence_token, token2group, patch_mask)

        if not return_aux:
            return out

        pred_idx = out.detach().argmax(dim=-1)
        per_patch_logits = rgc_aux["per_patch_logits"]
        clean_logits = rgc_aux.get("evidence_logits_clean", per_patch_logits)
        attr_weight = rgc_aux.get("patch_importance_for_attr", rgc_aux["patch_importance"])
        clean_patch_contribution = clean_logits * attr_weight.unsqueeze(-1)
        gather_idx = pred_idx.view(-1, 1, 1).expand(-1, clean_logits.size(1), 1)
        pred_forward_contrib = rgc_aux["patch_contribution"].gather(dim=-1, index=gather_idx).squeeze(-1)
        pred_clean_contrib = clean_patch_contribution.gather(dim=-1, index=gather_idx).squeeze(-1)
        pred_pos_importance = self._normalize_importance(torch.relu(pred_clean_contrib), patch_mask)
        pred_abs_importance = self._normalize_importance(pred_clean_contrib.abs(), patch_mask)
        patch_span = torch.stack(
            [
                patch_meta["effective_start"].to(device=x_enc.device),
                patch_meta["effective_end"].to(device=x_enc.device),
            ],
            dim=-1,
        )

        aux_out = {
            "binding_mode": "local_evidence_composition_cr",
            "tokens": tokens,
            "evidence_token": evidence_token,
            "query": query,
            "patch_meta": patch_meta,
            "patch_span": patch_span,
            "patch_padding_mask": patch_mask,
            "clean_patch_contribution": clean_patch_contribution,
            "pred_class_patch_contrib": pred_clean_contrib,
            "signed_patch_contribution": pred_clean_contrib,
            "pred_forward_patch_contrib": pred_forward_contrib,
            "forward_signed_patch_contribution": pred_forward_contrib,
            "pred_pos_importance": pred_pos_importance,
            "pred_abs_importance": pred_abs_importance,
        }
        aux_out.update(mvre_aux)
        aux_out.update(router_aux)
        aux_out.update(rgc_aux)
        return out, aux_out


AnchorMoE_CR = Model
