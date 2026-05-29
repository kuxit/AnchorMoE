"""Camera-ready AnchorMoE loss.

Aligned objective:
    L = CE + lambda_orth * L_orth + lambda_conf * L_conf

"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class NewModelLoss(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.label_smoothing = float(getattr(args, "label_smoothing", 0.0))
        self.loss_type = str(getattr(args, "loss_type", "ce")).lower()
        self.focal_gamma = float(getattr(args, "focal_gamma", 2.0))
        self.class_weight_mode = str(getattr(args, "class_weight_mode", "none")).lower()
        self.class_weight_values = getattr(args, "class_weights", None)
        self.class_counts = getattr(args, "class_counts", None)
        self.ce_loss = self._ce_for_eval

        self.lambda_orth_base = float(getattr(args, "anchor_div_lambda", 0.05))
        self.lambda_conf_base = float(getattr(args, "conf_lambda", 0.01))
        self.use_confidence = bool(getattr(args, "use_confidence", True))
        self.orth_warmup_epochs = int(getattr(args, "orth_warmup_epochs", 0))
        self.conf_warmup_epochs = int(getattr(args, "conf_warmup_epochs", 0))

        self.current_epoch = 0
        self.total_epochs = int(getattr(args, "train_epochs", 1))
        self.last_val_acc = None
        self.recent_train_ce = None

    def _class_weight(self, device, dtype):
        if self.class_weight_mode == "none":
            return None
        if self.class_weight_values is not None:
            w = torch.tensor(self.class_weight_values, device=device, dtype=dtype)
            return w / w.mean().clamp_min(1e-6)
        if self.class_weight_mode == "balanced" and self.class_counts is not None:
            counts = torch.tensor(self.class_counts, device=device, dtype=dtype).clamp_min(1.0)
            w = counts.sum() / (counts.numel() * counts)
            return w / w.mean().clamp_min(1e-6)
        return None

    def _ce_for_eval(self, logits, targets):
        targets = self._prepare_targets(targets)
        return F.cross_entropy(
            logits,
            targets,
            weight=self._class_weight(logits.device, logits.dtype),
            label_smoothing=self.label_smoothing,
        )

    def _classification_loss(self, logits, targets):
        targets = self._prepare_targets(targets)
        weight = self._class_weight(logits.device, logits.dtype)
        ce = F.cross_entropy(
            logits,
            targets,
            weight=weight,
            label_smoothing=self.label_smoothing,
            reduction="none" if self.loss_type == "focal" else "mean",
        )
        if self.loss_type != "focal":
            return ce
        pt = torch.softmax(logits, dim=-1).gather(1, targets.view(-1, 1)).squeeze(1).clamp(1e-6, 1.0)
        return (((1.0 - pt) ** self.focal_gamma) * ce).mean()

    def set_epoch(self, epoch: int, total_epochs=None, last_val_acc=None, recent_train_ce=None):
        self.current_epoch = int(epoch)
        if total_epochs is not None:
            self.total_epochs = int(total_epochs)
        self.last_val_acc = None if last_val_acc is None else float(last_val_acc)
        self.recent_train_ce = None if recent_train_ce is None else float(recent_train_ce)

    @staticmethod
    def _linear_warmup(base: float, current_epoch: int, warmup_epochs: int) -> float:
        if base <= 0:
            return 0.0
        if warmup_epochs <= 0:
            return float(base)
        ratio = min(1.0, max(0.0, float(current_epoch + 1) / float(warmup_epochs)))
        return float(base) * ratio

    def get_phase_info(self):
        return {
            "phase": "camera_ready",
            "epoch": self.current_epoch,
            "total_epochs": self.total_epochs,
            "lambda_orth_eff": self._linear_warmup(
                self.lambda_orth_base, self.current_epoch, self.orth_warmup_epochs
            ),
            "lambda_conf_eff": self._linear_warmup(
                self.lambda_conf_base, self.current_epoch, self.conf_warmup_epochs
            ),
        }

    @staticmethod
    def _unpack_outputs(outputs):
        logits = outputs
        aux = {}
        if isinstance(outputs, (tuple, list)):
            logits = outputs[0]
            if len(outputs) > 1 and isinstance(outputs[1], dict):
                aux = outputs[1]
        return logits, aux

    @staticmethod
    def _prepare_targets(targets):
        if targets.dtype != torch.long:
            targets = targets.long()
        if targets.dim() > 1:
            targets = targets.view(-1)
        return targets

    @staticmethod
    def _normalize_per_sample(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        x = x.masked_fill(~mask, 0.0)
        xmax = x.max(dim=1, keepdim=True).values.clamp_min(eps)
        return (x / xmax).clamp(0.0, 1.0) * mask.float()

    def _confidence_alignment_target(self, aux, targets, logits=None):
        """Build a detached patch-level confidence target.

        The target is the absolute predicted-class local evidence contribution
        computed before reliability gating and normalized per sample.
        """
        patch_contrib = aux.get("confidence_free_patch_contribution", None)
        if not torch.is_tensor(patch_contrib):
            patch_contrib = aux.get("clean_patch_contribution", None)
        if not torch.is_tensor(patch_contrib):
            patch_contrib = aux.get("patch_contribution", None)
        patch_mask = aux.get("patch_padding_mask", None)
        if not torch.is_tensor(patch_contrib):
            return None
        if torch.is_tensor(patch_mask):
            mask = patch_mask.bool()
        else:
            mask = torch.ones(patch_contrib.shape[:2], device=patch_contrib.device, dtype=torch.bool)

        with torch.no_grad():
            if torch.is_tensor(logits):
                full_logits = logits.detach()
            else:
                full_logits = patch_contrib.detach().sum(dim=1)
            pred_idx = full_logits.argmax(dim=-1)
            pred_drop = patch_contrib.detach().gather(
                dim=-1,
                index=pred_idx.view(-1, 1, 1).expand(-1, patch_contrib.size(1), 1),
            ).squeeze(-1)
            drop_score = pred_drop.abs()
            target = self._normalize_per_sample(drop_score, mask)
        return target

    def _confidence_alignment_loss(self, aux, targets, importance_gt=None, logits=None):
        if not self.use_confidence:
            return None, {}
        aggregated_conf = aux.get("patch_confidence", None)
        patch_mask = aux.get("patch_padding_mask", None)
        if not torch.is_tensor(aggregated_conf):
            return None, {}
        if torch.is_tensor(patch_mask):
            mask = patch_mask.bool()
        else:
            mask = torch.ones_like(aggregated_conf, dtype=torch.bool)

        if torch.is_tensor(importance_gt):
            target = importance_gt.to(device=aggregated_conf.device, dtype=aggregated_conf.dtype)
            if target.shape != aggregated_conf.shape:
                target = target.view_as(aggregated_conf)
            target = target.clamp(0.0, 1.0).detach()
        else:
            target = self._confidence_alignment_target(aux, targets, logits=logits)
        if target is None:
            return None, {}

        pred = aggregated_conf.clamp(1e-6, 1.0 - 1e-6)
        loss = F.binary_cross_entropy(pred[mask], target[mask]) if mask.any() else pred.new_tensor(0.0)
        info = {
            "conf_align": float(loss.detach().item()),
            "conf_target_mean": float(target[mask].detach().mean().item()) if mask.any() else 0.0,
            "conf_pred_mean": float(pred[mask].detach().mean().item()) if mask.any() else 0.0,
        }
        return loss, info

    @staticmethod
    def _diagnostics(aux, targets):
        info = {}
        patch_mask = aux.get("patch_padding_mask", None)
        mask = patch_mask.bool() if torch.is_tensor(patch_mask) else None

        evidence_weight = aux.get("evidence_weight", None)
        if torch.is_tensor(evidence_weight):
            if mask is None:
                mask = torch.ones_like(evidence_weight, dtype=torch.bool)
            ew = evidence_weight.masked_fill(~mask, 0.0)
            info["evidence_weight_max"] = float(ew.max(dim=1).values.detach().mean().item())
            ent = -(ew.clamp_min(1e-9).log() * ew)
            info["evidence_weight_entropy"] = float(
                ent.sum(dim=1).div(mask.float().sum(dim=1).clamp_min(1.0)).detach().mean().item()
            )

        aggregated_conf = aux.get("patch_confidence", None)
        if torch.is_tensor(aggregated_conf):
            if mask is None:
                mask = torch.ones_like(aggregated_conf, dtype=torch.bool)
            vals = aggregated_conf[mask]
            if vals.numel() > 0:
                info["confidence_mean"] = float(vals.detach().mean().item())
                info["confidence_min"] = float(vals.detach().min().item())
                info["confidence_max"] = float(vals.detach().max().item())

        routing_entropy = aux.get("routing_entropy", None)
        if torch.is_tensor(routing_entropy):
            info["routing_entropy"] = float(routing_entropy.detach().item())

        spectral_gate = aux.get("spectral_gate", None)
        if torch.is_tensor(spectral_gate):
            info["spectral_gate"] = float(spectral_gate.detach().mean().item())

        expert_usage = aux.get("expert_usage", None)
        if torch.is_tensor(expert_usage):
            info["expert_usage_max"] = float(expert_usage.detach().max().item())
            info["expert_usage_min"] = float(expert_usage.detach().min().item())

        return info

    def forward(self, outputs, targets, batch_x=None, padding_mask=None,
                importance_gt=None, model=None):
        logits, aux = self._unpack_outputs(outputs)
        targets = self._prepare_targets(targets)

        cls_loss = self._classification_loss(logits, targets)
        total_loss = cls_loss
        loss_dict = {"ce": float(cls_loss.detach().item())}

        lambda_orth = self._linear_warmup(
            self.lambda_orth_base, self.current_epoch, self.orth_warmup_epochs
        )
        lambda_conf = self._linear_warmup(
            self.lambda_conf_base, self.current_epoch, self.conf_warmup_epochs
        )
        loss_dict["lambda_orth_eff"] = float(lambda_orth)
        loss_dict["lambda_conf_eff"] = float(lambda_conf)

        orth_loss = aux.get("anchor_div_loss", None)
        if torch.is_tensor(orth_loss):
            loss_dict["orth"] = float(orth_loss.detach().item())
            if lambda_orth > 0:
                total_loss = total_loss + lambda_orth * orth_loss

        conf_loss, conf_info = self._confidence_alignment_loss(
            aux, targets, importance_gt=importance_gt, logits=logits
        )
        if torch.is_tensor(conf_loss):
            loss_dict.update(conf_info)
            if lambda_conf > 0:
                total_loss = total_loss + lambda_conf * conf_loss

        loss_dict.update(self._diagnostics(aux, targets))
        loss_dict["total"] = float(total_loss.detach().item())
        return total_loss, loss_dict


Loss = NewModelLoss
