import torch
import torch.nn as nn

from utils.patch_alignment import patchwise_l2_from_grad


class NewModelLoss(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.ce_loss = nn.CrossEntropyLoss()
        self.lambda_orth = float(getattr(args, "anchor_div_lambda", 0.0))
        self.lambda_conf = float(getattr(args, "conf_lambda", 0.0))
        self.current_epoch = 0

    def set_epoch(self, epoch: int):
        self.current_epoch = int(epoch)

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
    def _normalize_patch_scores(scores, valid_mask):
        scores = scores.float() * valid_mask
        return scores / scores.sum(dim=1, keepdim=True).clamp_min(1e-6)

    def forward(self, outputs, targets, batch_x=None, padding_mask=None, importance_gt=None, model=None):
        logits, aux = self._unpack_outputs(outputs)
        if targets.dtype != torch.long:
            targets = targets.long()
        if targets.dim() > 1:
            targets = targets.view(-1)

        loss_dict = {}
        cls_loss = self.ce_loss(logits, targets)
        total_loss = cls_loss
        loss_dict["ce"] = float(cls_loss.item())

        orth_loss = aux.get("anchor_div_loss", None)
        if torch.is_tensor(orth_loss):
            loss_dict["orth"] = float(orth_loss.item())
            if self.lambda_orth > 0:
                total_loss = total_loss + self.lambda_orth * orth_loss

        if self.lambda_conf > 0:
            x_raw = aux.get("x_raw", None)
            bar_kappa = aux.get("aggregated_confidence", None)
            patch_meta = aux.get("patch_meta", None)
            patch_mask = aux.get("patch_padding_mask", None)
            if not torch.is_tensor(x_raw):
                raise RuntimeError("loss_paper requires aux['x_raw'] for gradient supervision")
            if not torch.is_tensor(bar_kappa):
                raise RuntimeError("loss_paper requires aux['aggregated_confidence']")
            if patch_meta is None:
                raise RuntimeError("loss_paper requires aux['patch_meta']")

            grad_x = torch.autograd.grad(
                outputs=cls_loss,
                inputs=x_raw,
                create_graph=True,
                retain_graph=True,
                allow_unused=False,
            )[0]
            grad_mag = patchwise_l2_from_grad(grad_x=grad_x, patch_meta=patch_meta)
            valid_mask = patch_mask.float() if patch_mask is not None else torch.ones_like(grad_mag)

            if patch_mask is not None:
                assert int(patch_mask.size(1)) == int(bar_kappa.size(1)), (
                    f"patch mask length {int(patch_mask.size(1))} must equal confidence length {int(bar_kappa.size(1))}"
                )
                assert int(grad_mag.size(1)) == int(bar_kappa.size(1)), (
                    f"gradient target length {int(grad_mag.size(1))} must equal confidence length {int(bar_kappa.size(1))}"
                )

            target_dist = self._normalize_patch_scores(grad_mag.abs(), valid_mask)
            pred_dist = self._normalize_patch_scores(bar_kappa, valid_mask)
            conf_loss = ((pred_dist - target_dist) ** 2 * valid_mask).sum() / valid_mask.sum().clamp_min(1.0)

            total_loss = total_loss + self.lambda_conf * conf_loss
            loss_dict["conf"] = float(conf_loss.item())
            loss_dict["grad_target_mean"] = float(target_dist.mean().item())
            loss_dict["kappa_mean"] = float(pred_dist.mean().item())

        return total_loss, loss_dict


Loss = NewModelLoss
