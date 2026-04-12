import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft

from utils.patch_alignment import build_patch_meta, unfold_with_patch_meta


def _masked_softmax(x, mask, dim=-1, eps=1e-9):
    if mask is None:
        return torch.softmax(x, dim=dim)
    x = x.masked_fill(~mask, float("-inf"))
    p = torch.softmax(x, dim=dim)
    p = torch.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
    return p / p.sum(dim=dim, keepdim=True).clamp_min(eps)


def orthogonal_diversity_loss(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    if (not torch.is_tensor(x)) or x.dim() != 3:
        device = x.device if torch.is_tensor(x) else None
        return torch.tensor(0.0, device=device)
    _, k, _ = x.shape
    if k <= 1:
        return x.new_tensor(0.0)
    xn = F.normalize(x, p=2, dim=-1, eps=eps)
    gram = torch.bmm(xn, xn.transpose(1, 2))
    eye = torch.eye(k, device=x.device, dtype=x.dtype).unsqueeze(0)
    off_diag_sq = ((gram - eye) * (1.0 - eye)) ** 2
    return (off_diag_sq.sum(dim=(1, 2)) / float(k * (k - 1))).mean()


class PatchEmbedding(nn.Module):
    def __init__(self, c_in, d_model, patch_len, stride, dropout=0.1):
        super().__init__()
        self.patch_len = int(patch_len)
        self.stride = int(stride)
        self.conv1 = nn.Conv1d(c_in, d_model, kernel_size=patch_len, stride=stride)
        self.conv2 = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        patch_meta = build_patch_meta(
            seq_len=int(x.size(1)),
            patch_len=self.patch_len,
            stride=self.stride,
            device=x.device,
        )
        x_pad = F.pad(x.permute(0, 2, 1), (int(patch_meta["pad_left"]), 0))
        out = self.conv1(x_pad).transpose(1, 2)
        if int(out.size(1)) != int(patch_meta["num_patches"]):
            raise AssertionError(
                f"Temporal patch count mismatch: {int(out.size(1))} vs {int(patch_meta['num_patches'])}"
            )
        out = self.act(self.norm1(out))
        identity = out
        out = self.norm2(self.conv2(out.transpose(1, 2)).transpose(1, 2))
        out = self.dropout(out + identity)
        return out, patch_meta


class TemporalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:x.size(1)].unsqueeze(0)


class SpectralEmbed(nn.Module):
    def __init__(self, c_in, d_model, patch_len, stride, num_freq_bands=4, dropout=0.1):
        super().__init__()
        self.patch_len = int(patch_len)
        self.stride = int(stride)
        self.num_freq_bands = int(num_freq_bands)
        self.proj = nn.Sequential(
            nn.Linear(c_in * self.num_freq_bands, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(self, x_raw, patch_meta):
        windows = unfold_with_patch_meta(x_raw, patch_meta)
        fft_out = torch.fft.rfft(windows, dim=-1)
        amplitude = torch.abs(fft_out)
        freq_dim = int(amplitude.size(-1))
        band_size = max(1, freq_dim // self.num_freq_bands)
        bands = []
        for i in range(self.num_freq_bands):
            s = i * band_size
            e = min((i + 1) * band_size, freq_dim)
            bands.append(amplitude[..., s:e].pow(2).sum(dim=-1))
        band_energies = torch.stack(bands, dim=-1)
        band_energies = torch.log1p(band_energies)
        band_energies = band_energies.permute(0, 2, 1, 3).flatten(2)
        if int(band_energies.size(1)) != int(patch_meta["num_patches"]):
            raise AssertionError(
                f"Spectral patch count mismatch: {int(band_energies.size(1))} vs {int(patch_meta['num_patches'])}"
            )
        return self.proj(band_energies)


class RouterPaper(nn.Module):
    def __init__(
        self,
        d_model,
        num_groups,
        patch_len,
        stride,
        c_in,
        dropout=0.1,
        num_freq_bands=4,
        spectral_weight=0.3,
        use_relevance_query=True,
    ):
        super().__init__()
        self.num_groups = int(num_groups)
        self.spectral_weight = float(spectral_weight)
        self.use_relevance_query = bool(use_relevance_query)
        self.time_encoder = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.spectral_encoder = SpectralEmbed(
            c_in=c_in,
            d_model=d_model,
            patch_len=patch_len,
            stride=stride,
            num_freq_bands=num_freq_bands,
            dropout=dropout,
        )
        self.router_head = nn.Linear(d_model + 1, num_groups)

    def forward(self, patches, x_raw, patch_meta, padding_mask=None, eps=1e-6):
        time_feat = self.time_encoder(patches)
        spectral_feat = self.spectral_encoder(x_raw, patch_meta)
        if int(time_feat.size(1)) != int(spectral_feat.size(1)):
            raise AssertionError(
                f"Temporal/spectral patch mismatch: {int(time_feat.size(1))} vs {int(spectral_feat.size(1))}"
            )

        if padding_mask is not None:
            mask_f = padding_mask.unsqueeze(-1).float()
            global_repr = (patches * mask_f).sum(dim=1, keepdim=True)
            global_repr = global_repr / mask_f.sum(dim=1, keepdim=True).clamp_min(eps)
        else:
            global_repr = patches.mean(dim=1, keepdim=True)

        if self.use_relevance_query:
            task_relevance = (patches * global_repr).sum(dim=-1, keepdim=True)
        else:
            task_relevance = torch.zeros(
                patches.size(0),
                patches.size(1),
                1,
                dtype=patches.dtype,
                device=patches.device,
            )
        combined_feat = torch.cat(
            [
                (1.0 - self.spectral_weight) * time_feat + self.spectral_weight * spectral_feat,
                task_relevance,
            ],
            dim=-1,
        )
        logits = self.router_head(combined_feat)
        token2group = _masked_softmax(
            logits,
            padding_mask.unsqueeze(-1) if padding_mask is not None else None,
            dim=-1,
        )
        return token2group, combined_feat, logits


class PatchLevelExpert(nn.Module):
    def __init__(self, d_model, dropout=0.1, use_confidence=True):
        super().__init__()
        self.processor = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )
        self.norm = nn.LayerNorm(d_model)
        self.confidence = None
        if use_confidence:
            self.confidence = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, 1),
                nn.Sigmoid(),
            )

    def forward(self, x):
        out = self.norm(x + self.processor(x))
        if self.confidence is None:
            conf = out.new_ones(out.size(0), out.size(1))
            return out, conf
        conf = self.confidence(out).squeeze(-1)
        return out * conf.unsqueeze(-1), conf


class ReviewBlockPaper(nn.Module):
    def __init__(
        self,
        d_model,
        num_groups,
        patch_len,
        stride,
        c_in,
        dropout=0.1,
        num_freq_bands=4,
        spectral_weight=0.3,
        use_confidence=True,
        use_relevance_query=True,
    ):
        super().__init__()
        self.num_groups = int(num_groups)
        self.router = RouterPaper(
            d_model=d_model,
            num_groups=num_groups,
            patch_len=patch_len,
            stride=stride,
            c_in=c_in,
            dropout=dropout,
            num_freq_bands=num_freq_bands,
            spectral_weight=spectral_weight,
            use_relevance_query=use_relevance_query,
        )
        self.experts = nn.ModuleList(
            [PatchLevelExpert(d_model, dropout=dropout, use_confidence=use_confidence) for _ in range(num_groups)]
        )

    def forward(self, patches, x_raw, patch_meta, padding_mask=None, return_aux=False, compute_div_loss=False):
        token2group, routing_query, routing_logits = self.router(
            patches=patches,
            x_raw=x_raw,
            patch_meta=patch_meta,
            padding_mask=padding_mask,
        )
        expert_outs = []
        confidences = []
        for expert in self.experts:
            out_k, conf_k = expert(patches)
            expert_outs.append(out_k)
            confidences.append(conf_k)
        expert_outs = torch.stack(expert_outs, dim=2)
        confidences = torch.stack(confidences, dim=2)

        mixed_feat = (token2group.unsqueeze(-1) * expert_outs).sum(dim=2)
        aggregated_confidence = (token2group * confidences).sum(dim=-1)

        anchors_time = None
        anchor_div_loss = None
        if compute_div_loss:
            routing_mass = token2group
            if padding_mask is not None:
                routing_mass = routing_mass * padding_mask.unsqueeze(-1).float()
            group_mass = routing_mass.sum(dim=1, keepdim=True).clamp_min(1e-6)
            anchors_time = torch.einsum("blk,bld->bkd", routing_mass, patches)
            anchors_time = anchors_time / group_mass.squeeze(1).unsqueeze(-1)
            anchor_div_loss = orthogonal_diversity_loss(anchors_time)

        if return_aux:
            aux = {
                "routing_query": routing_query,
                "routing_logits": routing_logits,
                "token2group": token2group,
                "confidences": confidences,
                "aggregated_confidence": aggregated_confidence,
                "expert_outputs": expert_outs,
            }
            if anchors_time is not None:
                aux["anchors_time"] = anchors_time
                aux["anchors_q"] = anchors_time
                aux["anchor_div_loss"] = anchor_div_loss
            return mixed_feat, aux

        if compute_div_loss:
            return mixed_feat, anchor_div_loss if anchor_div_loss is not None else mixed_feat.new_tensor(0.0)
        return mixed_feat


class Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.task_name = args.task_name
        self.d_model = int(args.d_model)
        self.num_groups = int(getattr(args, "num_groups", 4))
        self.num_class = int(getattr(args, "num_class", 10))
        self.patch_len = int(getattr(args, "patch_len", 16))
        self.stride = int(getattr(args, "stride", 8))
        self.anchor_div_lambda = float(getattr(args, "anchor_div_lambda", 0.0))
        self.conf_lambda = float(getattr(args, "conf_lambda", 0.0))
        self.num_freq_bands = int(getattr(args, "num_freq_bands", 4))
        self.spectral_weight = float(getattr(args, "spectral_weight", 0.3))
        self.use_relevance_query = bool(getattr(args, "use_relevance_query", True))
        self.force_loss_aux = bool(self.anchor_div_lambda > 0 or self.conf_lambda > 0)

        self.patch_embed = PatchEmbedding(args.enc_in, self.d_model, self.patch_len, self.stride, args.dropout)
        self.pos_embed = TemporalPositionalEncoding(self.d_model)
        self.block = ReviewBlockPaper(
            d_model=self.d_model,
            num_groups=self.num_groups,
            patch_len=self.patch_len,
            stride=self.stride,
            c_in=args.enc_in,
            dropout=args.dropout,
            num_freq_bands=self.num_freq_bands,
            spectral_weight=self.spectral_weight,
            use_confidence=bool(getattr(args, "use_confidence", True)),
            use_relevance_query=self.use_relevance_query,
        )
        self.head = nn.Linear(self.d_model, self.num_class)

    @staticmethod
    def _build_patch_padding_mask(x_mark_enc, patch_meta, device):
        if x_mark_enc is None:
            return None
        pm = x_mark_enc.bool() if x_mark_enc.dtype == torch.bool else (x_mark_enc > 0.5)
        effective_end = patch_meta["effective_end"].to(device=device)
        effective_start = patch_meta["effective_start"].to(device=device)
        lengths = pm.sum(dim=1).to(torch.long)
        valid = (effective_end.unsqueeze(0) > effective_start.unsqueeze(0))
        valid = valid & (effective_start.unsqueeze(0) < lengths.unsqueeze(1))
        return valid

    def compute_patch_importance(self, token2group, confidences, patch_padding_mask=None):
        importance = (token2group * confidences).sum(dim=-1)
        if patch_padding_mask is not None:
            importance = importance * patch_padding_mask.float()
        return importance / importance.sum(dim=1, keepdim=True).clamp_min(1e-6)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None, return_aux=False):
        return_aux = bool(return_aux or (self.training and self.force_loss_aux))
        need_input_grad = self.training and self.conf_lambda > 0 and return_aux
        x_raw = x_enc.detach().requires_grad_(True) if need_input_grad else x_enc

        patches, patch_meta = self.patch_embed(x_raw)
        patches = self.pos_embed(patches)
        patch_padding_mask = self._build_patch_padding_mask(x_mark_enc, patch_meta, x_raw.device)
        if patch_padding_mask is None:
            patch_padding_mask = torch.ones(
                x_raw.size(0),
                int(patch_meta["num_patches"]),
                device=x_raw.device,
                dtype=torch.bool,
            )

        compute_div = self.training and self.anchor_div_lambda > 0
        if return_aux:
            feat, layer_aux = self.block(
                patches=patches,
                x_raw=x_raw,
                patch_meta=patch_meta,
                padding_mask=patch_padding_mask,
                return_aux=True,
                compute_div_loss=compute_div,
            )
            aux_layers = [layer_aux]
        else:
            if compute_div:
                feat, _ = self.block(
                    patches=patches,
                    x_raw=x_raw,
                    patch_meta=patch_meta,
                    padding_mask=patch_padding_mask,
                    return_aux=False,
                    compute_div_loss=True,
                )
            else:
                feat = self.block(
                    patches=patches,
                    x_raw=x_raw,
                    patch_meta=patch_meta,
                    padding_mask=patch_padding_mask,
                    return_aux=False,
                    compute_div_loss=False,
                )
            aux_layers = None

        patch_mask_f = patch_padding_mask.unsqueeze(-1).float()
        feat = feat * patch_mask_f
        patch_logits = self.head(feat)
        patch_logits = patch_logits * patch_mask_f
        out = patch_logits.sum(dim=1)

        if not return_aux:
            return out

        last_aux = aux_layers[-1]
        token2group = last_aux["token2group"]
        confidences = last_aux["confidences"]
        aggregated_confidence = last_aux["aggregated_confidence"]
        patch_importance = self.compute_patch_importance(
            token2group=token2group,
            confidences=confidences,
            patch_padding_mask=patch_padding_mask,
        )

        aux_out = {
            "layers": aux_layers,
            "final_feat": feat,
            "patches": patches,
            "patch_meta": patch_meta,
            "patch_padding_mask": patch_padding_mask,
            "token2group": token2group,
            "confidences": confidences,
            "aggregated_confidence": aggregated_confidence,
            "patch_importance": patch_importance,
            "patch_logits": patch_logits,
            "routing_query": last_aux["routing_query"],
            "x_raw": x_raw,
        }
        if "anchor_div_loss" in last_aux:
            aux_out["anchor_div_loss"] = last_aux["anchor_div_loss"]
        if "anchors_time" in last_aux:
            aux_out["anchors_time"] = last_aux["anchors_time"]
            aux_out["anchors_q"] = last_aux["anchors_q"]
        return out, aux_out


AnchorMoE = Model
