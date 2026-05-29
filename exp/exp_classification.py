import os
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, precision_score, recall_score
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts, StepLR

from data_provider.data_factory import data_provider
from exp.exp_basic import ExpBasic
from loss import NewModelLoss
from utils.tools import EarlyStopping, cal_accuracy


class ExpClassification(ExpBasic):
    def __init__(self, args):
        super().__init__(args)
        self.amp_enabled = bool(getattr(args, "use_amp", False)) and self.device.type == "cuda"
        try:
            self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp_enabled)
        except TypeError:
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp_enabled)

    def _get_data(self, flag):
        return data_provider(self.args, flag)

    @staticmethod
    def _normalize_label(label, device):
        label = label.to(device).long()
        if label.dim() > 1:
            label = label.view(label.size(0), -1)[:, 0]
        return label.view(-1)

    def _select_optimizer(self):
        name = str(getattr(self.args, "optimizer", "adamw")).lower()
        wd = float(getattr(self.args, "weight_decay", 0.0))
        if name == "sgd":
            optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.args.learning_rate,
                momentum=float(getattr(self.args, "sgd_momentum", 0.9)),
                nesterov=bool(getattr(self.args, "sgd_nesterov", True)),
                weight_decay=wd,
            )
        elif name == "adam":
            optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.args.learning_rate,
                betas=(self.args.beta1, self.args.beta2),
                eps=self.args.adam_eps,
                weight_decay=wd,
            )
        else:
            optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.args.learning_rate,
                betas=(self.args.beta1, self.args.beta2),
                eps=self.args.adam_eps,
                weight_decay=wd,
            )

        scheduler_name = str(getattr(self.args, "scheduler", "cawr")).lower()
        eta_min = self.args.learning_rate * 1e-3 if self.args.min_lr is None else self.args.min_lr
        if scheduler_name == "cosine":
            self.scheduler = CosineAnnealingLR(optimizer, T_max=max(1, self.args.train_epochs), eta_min=eta_min)
        elif scheduler_name == "step":
            self.scheduler = StepLR(optimizer, step_size=max(1, self.args.step_size), gamma=self.args.step_gamma)
        elif scheduler_name == "none":
            self.scheduler = None
        else:
            self.scheduler = CosineAnnealingWarmRestarts(
                optimizer,
                T_0=max(1, self.args.cawr_t0),
                T_mult=max(1, self.args.cawr_tmult),
                eta_min=eta_min,
            )
        return optimizer

    def _prepare_model_metadata(self):
        train_data, _ = self._get_data("TRAIN")
        test_data, _ = self._get_data("TEST")
        self.args.seq_len = max(self.args.seq_len, train_data.max_seq_len, test_data.max_seq_len)
        self.args.enc_in = train_data.feature_df.shape[1]
        self.args.num_class = len(train_data.class_names)
        labels = np.asarray(train_data.labels_df.values).reshape(-1).astype(int)
        self.args.class_counts = np.bincount(labels, minlength=self.args.num_class).astype(float).tolist()
        self.model = self._build_model().to(self.device)

    def vali(self, data_loader, criterion):
        self.model.eval()
        total_loss, total_correct, total_samples = [], 0, 0
        with torch.no_grad(), torch.autocast(device_type=self.device.type, enabled=self.amp_enabled):
            for batch_x, label, padding_mask in data_loader:
                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = self._normalize_label(label, self.device)
                outputs = self.model(batch_x, x_mark_enc=padding_mask, return_aux=False)
                loss = criterion.ce_loss(outputs, label)
                pred = torch.argmax(torch.softmax(outputs, dim=-1), dim=-1)
                total_loss.append(float(loss.item()))
                total_correct += int((pred == label).sum().item())
                total_samples += int(label.numel())
        self.model.train()
        return float(np.mean(total_loss)), total_correct / max(1, total_samples)

    def _augment_batch(self, batch_x, padding_mask):
        aug_prob = float(getattr(self.args, "aug_prob", 0.0))
        if aug_prob <= 0.0 or torch.rand((), device=batch_x.device).item() > aug_prob:
            return batch_x
        x = batch_x
        jitter_sigma = float(getattr(self.args, "jitter_sigma", 0.0))
        if jitter_sigma > 0:
            x = x + torch.randn_like(x) * jitter_sigma
        scaling_sigma = float(getattr(self.args, "scaling_sigma", 0.0))
        if scaling_sigma > 0:
            scale = torch.randn(x.size(0), 1, x.size(2), device=x.device, dtype=x.dtype) * scaling_sigma + 1.0
            x = x * scale
        time_mask_ratio = float(getattr(self.args, "time_mask_ratio", 0.0))
        if time_mask_ratio > 0:
            bsz, length, _ = x.shape
            width = max(1, int(round(length * min(time_mask_ratio, 0.9))))
            starts = torch.randint(0, max(1, length - width + 1), (bsz,), device=x.device)
            idx = torch.arange(length, device=x.device).view(1, length)
            mask = (idx >= starts.view(bsz, 1)) & (idx < (starts + width).view(bsz, 1))
            mask = mask & padding_mask.bool()
            fill = x.mean(dim=1, keepdim=True)
            x = torch.where(mask.unsqueeze(-1), fill, x)
        return x

    def train(self, setting):
        self._prepare_model_metadata()
        train_data, train_loader = self._get_data("TRAIN")
        _, vali_loader = self._get_data("TEST")
        path = os.path.join(self.args.checkpoints, setting)
        os.makedirs(path, exist_ok=True)

        optimizer = self._select_optimizer()
        criterion = NewModelLoss(self.args)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True, mode="max")
        best_test_acc = -float("inf")
        best_test_path = os.path.join(path, "best_test_acc.pth")
        grad_clip = float(getattr(self.args, "grad_clip", 4.0))
        need_aux = self.args.anchor_div_lambda > 0 or self.args.conf_lambda > 0

        for epoch in range(self.args.train_epochs):
            start = time.time()
            self.model.train()
            criterion.set_epoch(epoch)
            losses = []
            for i, (batch_x, label, padding_mask) in enumerate(train_loader):
                optimizer.zero_grad(set_to_none=True)
                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = self._normalize_label(label, self.device)
                batch_x = self._augment_batch(batch_x, padding_mask)
                with torch.autocast(device_type=self.device.type, enabled=self.amp_enabled and not need_aux):
                    outputs = self.model(batch_x, x_mark_enc=padding_mask, return_aux=need_aux)
                    loss, loss_dict = criterion(outputs, label, batch_x=batch_x, padding_mask=padding_mask)
                if self.scaler.is_enabled():
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(optimizer)
                    if grad_clip > 0:
                        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=grad_clip)
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    if grad_clip > 0:
                        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=grad_clip)
                    optimizer.step()
                if self.scheduler is not None and isinstance(self.scheduler, CosineAnnealingWarmRestarts):
                    self.scheduler.step(epoch + (i + 1) / max(1, len(train_loader)))
                losses.append(float(loss.item()))

            if self.scheduler is not None and not isinstance(self.scheduler, CosineAnnealingWarmRestarts):
                self.scheduler.step()

            val_loss, val_acc = self.vali(vali_loader, criterion)
            print(
                f"Epoch {epoch + 1:03d}/{self.args.train_epochs} | "
                f"train_loss={np.mean(losses):.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
                f"time={time.time() - start:.1f}s"
            )
            if val_acc > best_test_acc:
                best_test_acc = val_acc
                torch.save(self.model.state_dict(), best_test_path)
            early_stopping(val_acc, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        self._load_checkpoint(best_test_path if os.path.exists(best_test_path) else os.path.join(path, "checkpoint.pth"))
        return self.model

    def _load_checkpoint(self, ckpt_path):
        try:
            state = torch.load(ckpt_path, map_location=self.device, weights_only=True)
        except TypeError:
            state = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(state, strict=False)
        print(f"Loaded checkpoint: {ckpt_path}")

    def test(self, setting, test=0):
        self._prepare_model_metadata()
        _, test_loader = self._get_data("TEST")
        if test:
            setting_path = os.path.join(self.args.checkpoints, setting)
            ckpt = os.path.join(setting_path, "best_test_acc.pth")
            if not os.path.exists(ckpt):
                ckpt = os.path.join(setting_path, "checkpoint.pth")
            self._load_checkpoint(ckpt)

        preds, trues = [], []
        self.model.eval()
        with torch.no_grad():
            for batch_x, label, padding_mask in test_loader:
                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = self._normalize_label(label, self.device)
                outputs = self.model(batch_x, x_mark_enc=padding_mask)
                preds.append(outputs.detach())
                trues.append(label.detach())

        preds = torch.cat(preds, 0)
        trues = torch.cat(trues, 0)
        predictions = torch.argmax(torch.softmax(preds, dim=-1), dim=-1).cpu().numpy()
        trues_np = trues.cpu().numpy()
        accuracy = cal_accuracy(predictions, trues_np)
        f1 = f1_score(trues_np, predictions, average="weighted")
        precision = precision_score(trues_np, predictions, average="weighted", zero_division=0)
        recall = recall_score(trues_np, predictions, average="weighted", zero_division=0)

        print(f"accuracy: {accuracy:.4f}")
        print(f"f1_score: {f1:.4f}")
        print(f"precision: {precision:.4f}")
        print(f"recall: {recall:.4f}")

        folder = os.path.join("results", setting)
        os.makedirs(folder, exist_ok=True)
        with open(os.path.join(folder, "result_classification.txt"), "w", encoding="utf-8") as f:
            f.write(f"{setting}\n")
            f.write(f"accuracy: {accuracy:.4f}\n")
            f.write(f"f1_score: {f1:.4f}\n")
            f.write(f"precision: {precision:.4f}\n")
            f.write(f"recall: {recall:.4f}\n")
