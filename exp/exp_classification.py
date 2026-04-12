import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, cal_accuracy
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score
import traceback
from tools.timewarp import piecewise_time_warp
import math

try:
    from exp.visualization import ModelVisualizer
    _VISUALIZER_IMPORT_ERROR = None
except Exception as exc:
    ModelVisualizer = None
    _VISUALIZER_IMPORT_ERROR = exc

warnings.filterwarnings('ignore')


def plot_loss_curves(train_losses, vali_losses, test_losses, out_dir):
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, 'b-', label='Train Loss')
    plt.plot(epochs, vali_losses, 'r-', label='Vali Loss')
    plt.plot(epochs, test_losses, 'g--', label='Test Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'loss_curve.png'), dpi=300)
    plt.close()
    print(f"[Visualization] Loss curve saved to {out_dir}/loss_curve.png")


class Exp_Classification(Exp_Basic):
    def __init__(self, args):
        self.training_stats = {
            'total_params': 0,
            'avg_memory_mb': 0.0,
            'memory_samples': [],
            'inference_times': [],
            'training_times': []
        }
        super(Exp_Classification, self).__init__(args)

        self.amp_enabled = bool(getattr(self.args, "amp", False)) and (
            self.device.type == "cuda")
        self.amp_dtype = torch.float16 if self.amp_enabled else torch.float32
        self.scaler = torch.cuda.amp.GradScaler(
            enabled=self.amp_enabled and self.amp_dtype == torch.float16)

        # 记录Loss
        self.train_loss_recorder = []
        self.vali_loss_recorder = []
        self.test_loss_recorder = []

    def _make_visualizer(self, model_eval):
        if ModelVisualizer is None:
            print(
                "[Visualization] Optional visualization dependencies are unavailable; "
                f"skipping visualization features. Root cause: {_VISUALIZER_IMPORT_ERROR}"
            )
            return None
        return ModelVisualizer(
            model=model_eval,
            device=self.device,
            enc_in=self.enc_in,
            fs=self.fs,
            class_names=getattr(self.args, "class_names", None),
        )

    def _maybe_run_posthoc_compare(self, visualizer, data_loader, out_dir):
        if visualizer is None:
            return
        if not bool(getattr(self.args, "posthoc_compare_eval", False)):
            return
        fn = getattr(visualizer, "run_patch_faithfulness_benchmark", None)
        if not callable(fn):
            return
        os.makedirs(out_dir, exist_ok=True)
        fn(
            data_loader=data_loader,
            out_dir=out_dir,
            dataset_name=str(getattr(self.args, "model_id", "")),
            model_name=str(getattr(self.args, "model", "")),
            methods=str(getattr(self.args, "posthoc_compare_methods", "intrinsic,grad,input_x_grad")),
            max_samples=int(getattr(self.args, "posthoc_compare_max_samples", 64)),
            max_batches=int(getattr(self.args, "importance_eval_max_batches", -1)),
            use_only_correct=True,
            min_conf=0.0,
        )

    def _maybe_run_aopc(self, visualizer, data_loader, out_dir):
        if visualizer is None:
            return
        if not bool(getattr(self.args, "aopc_eval", False)):
            return
        fn = getattr(visualizer, "run_aopc_eval", None)
        if not callable(fn):
            return
        os.makedirs(out_dir, exist_ok=True)
        fn(
            data_loader=data_loader,
            out_dir=out_dir,
            dataset_name=str(getattr(self.args, "model_id", "")),
            model_name=str(getattr(self.args, "model", "")),
            max_samples=int(getattr(self.args, "aopc_max_samples", -1)),
            max_batches=int(getattr(self.args, "importance_eval_max_batches", -1)),
            use_only_correct=True,
            min_conf=0.0,
            perturb_mode=str(getattr(self.args, "aopc_perturb_mode", "zero")),
        )

    def _maybe_run_iou(self, visualizer, data_loader, out_dir):
        if visualizer is None:
            return
        if not bool(getattr(self.args, "iou_eval", False)):
            return
        fn = getattr(visualizer, "run_iou_eval", None)
        if not callable(fn):
            return
        os.makedirs(out_dir, exist_ok=True)
        fn(
            data_loader=data_loader,
            out_dir=out_dir,
            dataset_name=str(getattr(self.args, "model_id", "")),
            model_name=str(getattr(self.args, "model", "")),
            max_samples=int(getattr(self.args, "iou_max_samples", -1)),
            max_batches=int(getattr(self.args, "importance_eval_max_batches", -1)),
            use_only_correct=True,
            min_conf=0.0,
        )

    def _build_model(self):
        torch.manual_seed(self.args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.args.seed)

        from data_provider.uea import collate_fn
        from torch.utils.data import DataLoader

        train_data, _ = data_provider(self.args, 'TRAIN')
        test_data, _ = data_provider(self.args, 'TEST')
        vali_data, _ = data_provider(self.args, 'TEST')

        dataset_max_seq_len = max(
            train_data.max_seq_len, test_data.max_seq_len, vali_data.max_seq_len)
        if hasattr(self.args, 'seq_len') and self.args.seq_len is not None:
            if self.args.seq_len < dataset_max_seq_len:
                print(
                    f"警告: 显式设置的seq_len={self.args.seq_len}小于数据集最大长度={dataset_max_seq_len}，使用数据集最大值避免截断")
                self.args.seq_len = dataset_max_seq_len
        else:
            self.args.seq_len = dataset_max_seq_len

        train_loader = DataLoader(
            train_data,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
            drop_last=False,
            collate_fn=lambda x: collate_fn(x, max_len=self.args.seq_len)
        )
        test_loader = DataLoader(
            test_data,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            drop_last=False,
            collate_fn=lambda x: collate_fn(x, max_len=self.args.seq_len)
        )

        self.args.pred_len = 0
        self.args.enc_in = train_data.feature_df.shape[1]
        self.args.num_class = len(train_data.class_names)
        self.args.class_names = train_data.class_names

        model = self.model_dict[self.args.model](self.args).float()

        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel()
                        for p in model.parameters() if p.requires_grad)
        frozen = total - trainable
        self.training_stats['total_params'] = total
        self.training_stats['trainable_params'] = trainable
        self.training_stats['frozen_params'] = frozen

        print(f"=== Model Params ===")
        print(f"Total     : {total:,}")
        print(f"Trainable : {trainable:,}")
        print(f"Frozen    : {frozen:,}")

        if getattr(self.args, "use_multi_gpu", False) and getattr(self.args, "use_gpu", False):
            model = nn.DataParallel(model, device_ids=getattr(
                self.args, "device_ids", None))

        self.enc_in = self.args.enc_in

        self.fs = None
        for ds in [train_data, test_data, vali_data]:
            if hasattr(ds, 'fs'):
                self.fs = float(getattr(ds, 'fs'))
                break
            if hasattr(ds, 'sampling_rate'):
                self.fs = float(getattr(ds, 'sampling_rate'))
                break
        if self.fs is None:
            self.fs = 1.0
            print(
                "[analysis] fs not provided; using normalized frequency (cycles/sample, Nyquist=0.5)")
        else:
            print(f"[analysis] sampling rate fs = {self.fs} Hz")

        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(
            self.model.parameters(),
            lr=self.args.learning_rate,
            betas=(0.9, getattr(self.args, 'beta2', 0.99))
        )
        t0 = int(getattr(self.args, "cawr_t0", 10))
        tm = int(getattr(self.args, "cawr_tmult", 2))
        eta_min = float(getattr(self.args, "min_lr",
                        self.args.learning_rate * 1e-3))
        self.scheduler = CosineAnnealingWarmRestarts(
            model_optim, T_0=max(1, t0), T_mult=max(1, tm), eta_min=eta_min
        )
        return model_optim

    def _select_criterion(self):
        import loss
        import importlib
        importlib.reload(loss)
        # 使用新的统一 Loss 类
        return loss.NewModelLoss(self.args)

    def _needs_train_aux(self):
        return bool(
            float(getattr(self.args, "anchor_div_lambda", 0.0)) > 0
            or float(getattr(self.args, "conf_lambda", 0.0)) > 0
            or bool(getattr(self.args, "log_conf_diagnostics", False))
        )

    def _resume_state_path(self, path: str) -> str:
        return os.path.join(path, "resume_state.pth")

    def _save_resume_state(self, path: str, epoch: int, model_optim, best_test_acc: float, early_stopping):
        state = {
            "epoch": int(epoch),
            "model": self.model.state_dict(),
            "optimizer": model_optim.state_dict(),
            "scheduler": self.scheduler.state_dict() if hasattr(self, "scheduler") and self.scheduler is not None else None,
            "scaler": self.scaler.state_dict() if self.scaler is not None else None,
            "best_test_acc": float(best_test_acc),
            "early_stopping_counter": int(getattr(early_stopping, "counter", 0)),
            "early_stopping_best_score": getattr(early_stopping, "best_score", None),
            "train_loss_recorder": list(getattr(self, "train_loss_recorder", [])),
            "vali_loss_recorder": list(getattr(self, "vali_loss_recorder", [])),
            "test_loss_recorder": list(getattr(self, "test_loss_recorder", [])),
            "training_stats": dict(getattr(self, "training_stats", {})),
        }
        torch.save(state, self._resume_state_path(path))

    def _maybe_load_resume_state(self, path: str, model_optim, early_stopping):
        if not bool(getattr(self.args, "resume", True)):
            return 0, -float("inf")
        resume_path = self._resume_state_path(path)
        if not os.path.exists(resume_path):
            return 0, -float("inf")

        state = torch.load(resume_path, map_location=self.device)
        model_state = state.get("model", None)
        if isinstance(model_state, dict):
            model_is_dp = isinstance(self.model, nn.DataParallel)
            keys = list(model_state.keys())
            has_module_prefix = (len(keys) > 0 and keys[0].startswith("module."))
            if model_is_dp and not has_module_prefix:
                model_state = {f"module.{k}": v for k, v in model_state.items()}
            if (not model_is_dp) and has_module_prefix:
                model_state = {k.replace("module.", "", 1): v for k, v in model_state.items()}
            self.model.load_state_dict(model_state, strict=False)
        if state.get("optimizer", None) is not None:
            model_optim.load_state_dict(state["optimizer"])
        if state.get("scheduler", None) is not None and hasattr(self, "scheduler") and self.scheduler is not None:
            self.scheduler.load_state_dict(state["scheduler"])
        if state.get("scaler", None) is not None and self.scaler is not None:
            self.scaler.load_state_dict(state["scaler"])

        self.train_loss_recorder = list(state.get("train_loss_recorder", []))
        self.vali_loss_recorder = list(state.get("vali_loss_recorder", []))
        self.test_loss_recorder = list(state.get("test_loss_recorder", []))
        if isinstance(state.get("training_stats", None), dict):
            self.training_stats.update(state["training_stats"])

        early_stopping.counter = int(state.get("early_stopping_counter", 0))
        early_stopping.best_score = state.get("early_stopping_best_score", None)
        start_epoch = int(state.get("epoch", -1)) + 1
        best_test_acc = float(state.get("best_test_acc", -float("inf")))
        print(f"[Resume] Loaded {resume_path}; resuming from epoch {start_epoch + 1}")
        return start_epoch, best_test_acc

    def _load_checkpoint_safely(self, ckpt_path: str):
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        ckpt = torch.load(ckpt_path, map_location=self.device)

        if isinstance(ckpt, dict):
            if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
                state = ckpt["state_dict"]
            elif "model" in ckpt and isinstance(ckpt["model"], dict):
                state = ckpt["model"]
            else:
                tensor_dict = {
                    k: v for k, v in ckpt.items() if isinstance(v, torch.Tensor)}
                state = tensor_dict if len(tensor_dict) > 0 else ckpt
        else:
            state = ckpt

        model_is_dp = isinstance(self.model, nn.DataParallel)
        keys = list(state.keys())
        has_module_prefix = (len(keys) > 0 and keys[0].startswith("module."))

        if model_is_dp and not has_module_prefix:
            state = {f"module.{k}": v for k, v in state.items()}
        if (not model_is_dp) and has_module_prefix:
            state = {k.replace("module.", "", 1): v for k, v in state.items()}

        missing, unexpected = self.model.load_state_dict(state, strict=False)
        if missing or unexpected:
            print("[WARN] load_state_dict strict=False:")
            if missing:
                print("  missing keys:", missing[:20], "..." if len(
                    missing) > 20 else "")
            if unexpected:
                print("  unexpected keys:", unexpected[:20], "..." if len(
                    unexpected) > 20 else "")

        print(f"[OK] Loaded checkpoint: {ckpt_path}")

    def visualize_from_checkpoint(self, ckpt_path: str, max_batches: int = 4, topk: int = 2):
        _, test_loader = self._get_data(flag='TEST')
        self._load_checkpoint_safely(ckpt_path)
        self.model.eval()

        setting = os.path.basename(os.path.dirname(ckpt_path))
        out_root = os.path.join(self.args.checkpoints, setting, "viz_only")
        os.makedirs(out_root, exist_ok=True)

        model_eval = self.model.module if isinstance(
            self.model, nn.DataParallel) else self.model
        visualizer = self._make_visualizer(model_eval)
        if visualizer is None:
            return

        out_dir_amp = os.path.join(out_root, "paper_ampcase")
        os.makedirs(out_dir_amp, exist_ok=True)
        if hasattr(model_eval, "inspect_aux"):
            print("[Visualization] 生成 paper_story_only（频率-语义对齐）...")
            visualizer.run_paper_story_only(
                test_loader,
                out_dir=out_dir_amp,
                max_batches=8,
                top_channels=4,
                max_stories=3,
            )
        else:
            print("[Visualization] 跳过 paper_story_only：当前模型未实现 inspect_aux。")

        out_dir_feat = os.path.join(out_root, "feature_viz")
        os.makedirs(out_dir_feat, exist_ok=True)
        print("[Visualization] feature viz (t-SNE)...")
        visualizer.run_feature_viz(
            test_loader,
            out_dir=out_dir_feat,
            take="prepool",
            hook_module=model_eval.norm
        )

        out_dir_vec = os.path.join(out_root, "vectorosc_aux")
        os.makedirs(out_dir_vec, exist_ok=True)
        print("[Visualization] VectorOsc aux viz (freq/amp/waves/harm_weights)...")
        try:
            from utils.viz_utils import plot_vectorosc_aux
        except Exception:
            plot_vectorosc_aux = None

        if plot_vectorosc_aux is not None:
            with torch.no_grad():
                for bi, batch in enumerate(test_loader):
                    if bi >= int(max_batches):
                        break

                    batch_x, label, padding_mask = batch[0], batch[1], batch[2] if len(
                        batch) > 2 else None
                    batch_x = batch_x.float().to(self.device)
                    padding_mask = padding_mask.float().to(
                        self.device) if padding_mask is not None else None
                    label = self._normalize_label(label, self.device)
                    if label.numel() == 0:
                        continue

                    aux_list = None
                    outputs = None
                    if hasattr(model_eval, 'forward_with_recon'):
                        outputs, _, aux_list = model_eval.forward_with_recon(
                            batch_x, x_mark_enc=padding_mask
                        )
                    else:
                        outputs = self.model(
                            batch_x, x_mark_enc=padding_mask, return_aux=True)
                        # 兼容 (logits, aux_dict) 返回
                        if isinstance(outputs, (tuple, list)) and len(outputs) >= 1:
                            outputs = outputs[0]

                    if outputs is None:
                        continue
                    if outputs.dim() == 3:
                        outputs = outputs.mean(dim=1)
                    pred = torch.argmax(outputs, dim=-1).view(-1)

                    y0 = int(label.view(-1)[0].item())
                    p0 = int(pred.view(-1)[0].item())
                    tag = f"b{bi}_y{y0}_pred{p0}"

                    if isinstance(aux_list, (list, tuple)) and len(aux_list) > 0:
                        plot_vectorosc_aux(
                            aux_list=aux_list,
                            out_dir=out_dir_vec,
                            sample_idx=0,
                            layers=[-1],
                            topk_heads=int(topk),
                            tag_prefix=tag,
                        )

        print(f"[OK] Visualization saved to: {out_root}")

    def _save_training_stats(self, setting):
        path = os.path.join(self.args.checkpoints, setting)
        os.makedirs(path, exist_ok=True)

        stats_file = os.path.join(path, "training_stats.txt")

        if len(self.training_stats['memory_samples']) > 0:
            avg_memory = np.mean(self.training_stats['memory_samples'])
            self.training_stats['avg_memory_mb'] = avg_memory
        else:
            avg_memory = 0.0

        avg_inference_time = np.mean(self.training_stats['inference_times']) if len(
            self.training_stats['inference_times']) > 0 else 0.0
        avg_training_time = np.mean(self.training_stats['training_times']) if len(
            self.training_stats['training_times']) > 0 else 0.0

        with open(stats_file, 'w', encoding='utf-8') as f:
            f.write("=== Training Statistics ===\n")
            f.write(
                f"Total parameters: {self.training_stats['total_params']:,}\n")
            f.write(f"Average GPU memory usage: {avg_memory:.2f} MB\n")
            f.write(
                f"Average inference time per iter: {avg_inference_time*1000:.4f} ms\n")
            f.write(
                f"Average training time per iter: {avg_training_time*1000:.4f} ms\n")

        print("=== Training Statistics ===")
        print(f"Total parameters: {self.training_stats['total_params']:,}")
        print(f"Average GPU memory usage: {avg_memory:.2f} MB")
        print(
            f"Average inference time per iter: {avg_inference_time*1000:.4f} ms")
        print(
            f"Average training time per iter: {avg_training_time*1000:.4f} ms")
        print(f"Training stats saved to: {stats_file}")

        if len(self.train_loss_recorder) > 0:
            plot_loss_curves(
                self.train_loss_recorder,
                self.vali_loss_recorder,
                self.test_loss_recorder,
                path
            )

    def _update_memory_stats(self):
        if torch.cuda.is_available():
            current_memory = torch.cuda.memory_allocated() / 1024**2
            self.training_stats['memory_samples'].append(current_memory)

    @torch.no_grad()
    def _collect_confidence_stats(self, loader, max_batches: int, conf_low_th: float):
        model_eval = self.model.module if isinstance(
            self.model, nn.DataParallel) else self.model
        model_eval.eval()

        conf_list = []
        for bi, batch in enumerate(loader):
            if bi >= int(max_batches):
                break
            batch_x, _, padding_mask = batch[0], batch[1], batch[2] if len(
                batch) > 2 else None
            batch_x = batch_x.float().to(self.device)
            padding_mask = padding_mask.float().to(
                self.device) if padding_mask is not None else None

            ret = model_eval(batch_x, x_mark_enc=padding_mask, return_aux=True)
            if not (isinstance(ret, (tuple, list)) and len(ret) == 2 and isinstance(ret[1], dict)):
                continue
            aux = ret[1]
            layers = aux.get("layers", None)
            if not isinstance(layers, list) or len(layers) == 0:
                continue
            last = layers[-1]
            if not isinstance(last, dict):
                continue
            conf = last.get("confidences", None)  # [B,K,1]
            if conf is None or (not torch.is_tensor(conf)) or conf.dim() != 3:
                continue
            conf_list.append(conf.detach().float().cpu())

        if len(conf_list) == 0:
            return None, None

        conf_all = torch.cat(conf_list, dim=0)  # [Btot,K,1]
        conf_all = conf_all.squeeze(-1)         # [Btot,K]
        conf_mean = conf_all.mean(dim=0)        # [K]
        conf_low_rate = (conf_all < float(conf_low_th)
                         ).float().mean(dim=0)  # [K]
        return conf_mean.numpy(), conf_low_rate.numpy()

    @staticmethod
    def _extract_expert_modules(model_eval):
        if hasattr(model_eval, "shared_experts"):
            return list(model_eval.shared_experts)
        return []

    def _collect_grad_norms(self, model_eval):
        experts = self._extract_expert_modules(model_eval)
        if len(experts) == 0:
            return None
        vals = []
        for ex in experts:
            s = 0.0
            for p in ex.parameters():
                if p.grad is None:
                    continue
                g = p.grad.detach()
                s += float(torch.norm(g, p=2).item())
            vals.append(s)
        return np.array(vals, dtype=np.float32)

    @staticmethod
    def _save_conf_risk_plots(out_dir: str, conf_means, conf_lows, grad_means):
        os.makedirs(out_dir, exist_ok=True)

        def _heatmap(arr, title, fname, vmin=None, vmax=None):
            plt.figure(figsize=(10, 3.6))
            plt.imshow(arr, aspect="auto", interpolation="nearest",
                       vmin=vmin, vmax=vmax, cmap="viridis")
            plt.colorbar()
            plt.title(title)
            plt.xlabel("expert (K)")
            plt.ylabel("epoch")
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, fname), dpi=300)
            plt.close()

        if conf_means is not None:
            _heatmap(conf_means, "Confidence mean (last layer)",
                     "conf_mean_heatmap.png", vmin=0.0, vmax=1.0)
        if conf_lows is not None:
            _heatmap(conf_lows, "Low-confidence rate (conf < th)",
                     "conf_lowrate_heatmap.png", vmin=0.0, vmax=1.0)
        if grad_means is not None:
            _heatmap(np.log10(grad_means + 1e-12),
                     "log10(grad norm) per expert", "grad_norm_log10_heatmap.png")

        try:
            plt.figure(figsize=(8, 4))
            if conf_means is not None:
                plt.plot(conf_means.mean(axis=1), label="mean(conf)")
            if conf_lows is not None:
                plt.plot(conf_lows.mean(axis=1), label="mean(low-rate)")
            plt.title("Expert silence risk trends")
            plt.xlabel("epoch")
            plt.ylabel("value")
            plt.grid(True, alpha=0.25)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "risk_trends.png"), dpi=300)
            plt.close()
        except Exception:
            pass

    @staticmethod
    def _normalize_label(label: torch.Tensor, device: torch.device) -> torch.Tensor:
        label = label.to(device).long()
        if label.dim() > 1:
            label = label.view(label.size(0), -1)[:, 0]
        return label.view(-1)

    def vali(self, vali_data, vali_loader, criterion):
        self.model.eval()
        total_loss = []
        total_correct = 0
        total_samples = 0

        autocast_ctx = torch.autocast(
            device_type=self.device.type,
            dtype=self.amp_dtype,
            enabled=self.amp_enabled
        )

        with torch.no_grad(), autocast_ctx:
            for i, (batch_x, label, padding_mask) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = self._normalize_label(label, self.device)

                if label.numel() == 0:
                    continue

                outputs = self.model(
                    batch_x, x_mark_enc=padding_mask, return_aux=False)

                if outputs.dim() == 3:
                    outputs = outputs.mean(dim=1)

                # Validation keeps only the classification term.
                loss = criterion.ce_loss(outputs, label)
                probs = torch.softmax(outputs, dim=1)
                predictions = torch.argmax(probs, dim=1).view(-1)
                correct = (predictions == label).sum().item()

                total_loss.append(float(loss.item()))
                total_correct += correct
                total_samples += label.size(0)

        if len(total_loss) == 0 or total_samples == 0:
            return 0.0, 0.0

        avg_loss = float(np.average(total_loss))
        accuracy = float(total_correct) / float(total_samples)

        self.model.train()
        return avg_loss, accuracy

    def train(self, setting):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        train_data, train_loader = self._get_data(flag='TRAIN')
        vali_data, vali_loader = self._get_data(flag='TEST')
        test_data, test_loader = self._get_data(flag='TEST')

        path = os.path.join(self.args.checkpoints, setting)
        os.makedirs(path, exist_ok=True)

        best_test_acc = -float("inf")
        best_test_path = os.path.join(path, "best_test_acc.pth")

        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(
            patience=self.args.patience,
            verbose=True,
            mode='max'
        )

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        start_epoch, resume_best_test_acc = self._maybe_load_resume_state(
            path, model_optim, early_stopping
        )
        if resume_best_test_acc > best_test_acc:
            best_test_acc = resume_best_test_acc
        need_train_aux = self._needs_train_aux()

        for epoch in range(start_epoch, self.args.train_epochs):
            epoch_start_time = time.time()
            iter_count = 0
            train_loss = []
            loss_terms_acc = {}

            self.model.train()
            if hasattr(criterion, "set_epoch"):
                criterion.set_epoch(epoch)

            grad_every = int(getattr(self.args, "grad_risk_every", 0))
            if grad_every <= 0:
                grad_every = max(1, train_steps // 10)
            grad_acc = None
            grad_n = 0

            for i, batch in enumerate(train_loader):
                if len(batch) == 4:
                    batch_x, label, padding_mask, importance_gt_batch = batch
                else:
                    batch_x, label, padding_mask = batch
                    importance_gt_batch = None

                iter_count += 1
                iter_start_time = time.time()
                model_optim.zero_grad(set_to_none=True)

                batch_x = batch_x.float().to(self.device)
                if (
                    float(getattr(self.args, "conf_lambda", 0.0)) > 0
                    or bool(getattr(self.args, "log_conf_diagnostics", False))
                ):
                    batch_x.requires_grad_(True)
                padding_mask = padding_mask.float().to(self.device)
                label = self._normalize_label(label, self.device)
                if importance_gt_batch is not None:
                    importance_gt_batch = importance_gt_batch.to(self.device)

                inference_start = time.time()
                with torch.autocast(
                    device_type=self.device.type,
                    dtype=self.amp_dtype,
                    enabled=self.amp_enabled and (not need_train_aux)
                ):
                    model_eval = self.model.module if isinstance(
                        self.model, nn.DataParallel) else self.model

                    if hasattr(model_eval, 'forward_with_recon'):
                        outputs, _, _ = model_eval.forward_with_recon(
                            batch_x, x_mark_enc=padding_mask
                        )
                    else:
                        outputs = self.model(
                            batch_x,
                            x_mark_enc=padding_mask,
                            return_aux=need_train_aux,
                        )

                    # 统一调用 Loss 计算
                    loss, loss_dict = criterion(
                        outputs=outputs,
                        targets=label,
                        batch_x=batch_x,
                        padding_mask=padding_mask,
                        importance_gt=importance_gt_batch,
                        model=self.model
                    )

                inference_time = time.time() - inference_start
                self.training_stats['inference_times'].append(inference_time)

                for k, v in loss_dict.items():
                    loss_terms_acc.setdefault(k, []).append(v)

                if self.scaler.is_enabled():
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(model_optim)
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=4.0)

                    if bool(getattr(self.args, "conf_risk_viz", False)) and ((i % grad_every) == 0):
                        g = self._collect_grad_norms(model_eval)
                        if g is not None:
                            grad_acc = g if grad_acc is None else (
                                grad_acc + g)
                            grad_n += 1
                    self.scaler.step(model_optim)
                    self.scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=4.0)
                    if bool(getattr(self.args, "conf_risk_viz", False)) and ((i % grad_every) == 0):
                        g = self._collect_grad_norms(model_eval)
                        if g is not None:
                            grad_acc = g if grad_acc is None else (
                                grad_acc + g)
                            grad_n += 1
                    model_optim.step()

                training_time = time.time() - iter_start_time
                self.training_stats['training_times'].append(training_time)
                self._update_memory_stats()

                try:
                    self.scheduler.step(epoch + (i + 1) / train_steps)
                except Exception:
                    pass

                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    terms_str = ", ".join(
                        [f"{k}:{np.mean(v):.6f}" for k, v in loss_terms_acc.items() if len(v) > 0])
                    print(
                        f"\titers: {i+1}, epoch: {epoch+1} | loss: {loss.item():.7f} | {terms_str}")
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * \
                        ((self.args.train_epochs - epoch) * train_steps - i)
                    print(
                        f"\tspeed: {speed:.4f}s/iter; left time: {left_time:.4f}s")
                    iter_count = 0
                    time_now = time.time()

            epoch_train_loss = float(np.average(train_loss)) if len(
                train_loss) > 0 else 0.0
            self.train_loss_recorder.append(epoch_train_loss)
            epoch_terms = {
                k: float(np.mean(v)) for k, v in loss_terms_acc.items() if len(v) > 0
            }

            epoch_duration = time.time() - epoch_start_time
            print("Epoch: {} cost time: {:.2f}s".format(
                epoch + 1, epoch_duration))
            if len(epoch_terms) > 0:
                terms_str = ", ".join([f"{k}:{v:.6f}" for k, v in epoch_terms.items()])
                print(f"Epoch: {epoch + 1} train diagnostics | {terms_str}")

            vali_loss, val_accuracy = self.vali(
                vali_data, vali_loader, criterion)
            test_loss, test_accuracy = self.vali(
                test_data, test_loader, criterion)

            self.vali_loss_recorder.append(vali_loss)
            self.test_loss_recorder.append(test_loss)

            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.3f} Vali Loss: {3:.3f} "
                "Vali Acc: {4:.3f} Test Loss: {5:.3f} Test Acc: {6:.3f}"
                .format(epoch + 1, train_steps, epoch_train_loss, vali_loss, val_accuracy, test_loss, test_accuracy)
            )

            try:
                if test_accuracy > best_test_acc:
                    best_test_acc = float(test_accuracy)
                    torch.save(self.model.state_dict(), best_test_path)
            except Exception as e:
                print(f"[WARN] saving best_test failed: {e}")

            try:
                self._save_resume_state(
                    path=path,
                    epoch=epoch,
                    model_optim=model_optim,
                    best_test_acc=best_test_acc,
                    early_stopping=early_stopping,
                )
            except Exception as e:
                print(f"[WARN] saving resume state failed: {e}")

            early_stopping(val_accuracy, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            if bool(getattr(self.args, "enable_viz", True)) and bool(getattr(self.args, "conf_risk_viz", False)):
                try:
                    if not hasattr(self, "_risk_conf_means"):
                        self._risk_conf_means = []
                        self._risk_conf_lows = []
                        self._risk_grad_means = []
                    conf_mean, conf_low = self._collect_confidence_stats(
                        train_loader,
                        max_batches=int(
                            getattr(self.args, "conf_risk_max_batches", 4)),
                        conf_low_th=float(
                            getattr(self.args, "conf_low_th", 0.1)),
                    )
                    if conf_mean is not None:
                        self._risk_conf_means.append(conf_mean)
                        self._risk_conf_lows.append(conf_low)
                    if grad_acc is not None and grad_n > 0:
                        self._risk_grad_means.append(
                            (grad_acc / float(grad_n)))
                except Exception:
                    pass

        self._update_memory_stats()
        self._save_training_stats(setting)

        best_candidates = [
            os.path.join(path, "best_test_acc.pth"),
            os.path.join(path, "checkpoint.pth"),
        ]
        loaded = False
        for ckpt in best_candidates:
            if os.path.exists(ckpt):
                try:
                    self._load_checkpoint_safely(ckpt)
                    loaded = True
                    break
                except Exception as e:
                    print(f"[WARN] failed to load {ckpt}: {e}")
        if not loaded:
            print("[WARN] No checkpoint loaded; using current in-memory weights.")

        try:
            model_eval = self.model.module if isinstance(
                self.model, nn.DataParallel) else self.model
            visualizer = self._make_visualizer(model_eval)
            is_router_model = bool(
                hasattr(model_eval, "use_router_system")
                or hasattr(model_eval, "shared_experts")
            )

            if visualizer is None:
                print("[Visualization] Skipping post-training visualization and importance analysis.")
                return self.model

            if not bool(getattr(self.args, "enable_viz", True)):
                if is_router_model and bool(getattr(self.args, "importance_eval", True)):
                    out_dir_imp = os.path.join(path, "importance_robustness")
                    os.makedirs(out_dir_imp, exist_ok=True)
                    try:
                        print(
                            "[Visualization] enable_viz=False，仅运行重要性鲁棒性评估（MoRF/LeRF）...")
                        fn4 = getattr(
                            visualizer, "run_importance_robustness_eval", None)
                        if callable(fn4):
                            fn4(
                                data_loader=test_loader,
                                out_dir=out_dir_imp,
                                ratios=None,
                                max_batches=int(
                                    getattr(self.args, "importance_eval_max_batches", -1)),
                                use_only_correct=True,
                                min_conf=float(
                                    getattr(self.args, "importance_eval_min_conf", 0.5)),
                                num_channels=int(
                                    getattr(self.args, "explain_patch_channels", 4)),
                                layer=-1,
                                noise_std_scale=0.5,
                            )
                        fn_gt = getattr(
                            visualizer, "run_importance_gt_correlation", None)
                        if callable(fn_gt) and hasattr(test_loader, "dataset") and hasattr(test_loader.dataset, "get_importance_gt_patch_level"):
                            fn_gt(test_loader, out_dir_imp, max_batches=10)
                    except Exception as e:
                        print(f"[WARN] importance_eval(light) 失败（可忽略）: {e}")
                else:
                    print("[Visualization] enable_viz=False，跳过训练后重可视化。")
                return self.model

            out_dir_amp = os.path.join(path, "paper_ampcase")
            os.makedirs(out_dir_amp, exist_ok=True)
            print("[Visualization] 生成 paper_story_only（仅证据级图）...")
            visualizer.run_paper_story_only(
                test_loader,
                out_dir=out_dir_amp,
                max_batches=8,
                top_channels=4,
                max_stories=2,
            )

            out_dir_feat = os.path.join(path, "feature_viz")
            os.makedirs(out_dir_feat, exist_ok=True)
            print("[Visualization] 生成 t-SNE...")
            hook_module = getattr(model_eval, "norm", None)
            if hook_module is None:
                hook_module = getattr(
                    model_eval, "layer_norm", None) or model_eval
            visualizer.run_feature_viz(
                test_loader,
                out_dir=out_dir_feat,
                take="prepool",
                hook_module=hook_module
            )

            if is_router_model:
                out_dir_attn = os.path.join(path, "router_attn_viz")
                os.makedirs(out_dir_attn, exist_ok=True)
                try:
                    print("[Visualization] 生成 router anchor-attention 可视化...")
                    visualizer.run_router_attn_viz(
                        test_loader,
                        out_dir=out_dir_attn,
                        max_batches=2,
                        sample_idx=0,
                        layers=[-1],
                    )
                except Exception as e:
                    print(f"[WARN] router_attn_viz 失败（可忽略）: {e}")

            if is_router_model and bool(getattr(self.args, "anchor_tsne_viz", True)):
                out_dir_tsne = os.path.join(path, "anchor_tsne")
                os.makedirs(out_dir_tsne, exist_ok=True)
                try:
                    print("[Visualization] 生成各层 anchors t-SNE 可视化...")
                    visualizer.run_anchor_tsne_viz(
                        test_loader,
                        out_dir=out_dir_tsne,
                        max_batches=int(
                            getattr(self.args, "anchor_tsne_max_batches", 3)),
                        perplexity=int(
                            getattr(self.args, "anchor_tsne_perplexity", 30)),
                        max_points=int(
                            getattr(self.args, "anchor_tsne_max_points", 2000)),
                    )
                except Exception as e:
                    print(f"[WARN] anchor_tsne_viz 失败（可忽略）: {e}")

            if is_router_model and bool(getattr(self.args, "explain_patch_viz", True)):
                out_dir_explain = os.path.join(path, "explain_patches")
                os.makedirs(out_dir_explain, exist_ok=True)
                try:
                    print("[Visualization] 生成最终决策高置信度 patch 可视化...")
                    fn = getattr(visualizer, "run_explain_patch_viz", None)
                    if callable(fn):
                        fn(
                            test_loader,
                            out_dir=out_dir_explain,
                            max_batches=int(
                                getattr(self.args, "explain_patch_max_batches", 2)),
                            max_samples=int(
                                getattr(self.args, "explain_patch_max_samples", 4)),
                            topk=int(
                                getattr(self.args, "explain_patch_topk", 6)),
                            num_channels=int(
                                getattr(self.args, "explain_patch_channels", 4)),
                            layer=-1,
                        )
                except Exception as e:
                    print(f"[WARN] explain_patch_viz 失败（可忽略）: {e}")

            if is_router_model and bool(getattr(self.args, "explain_contrast_viz", True)):
                out_dir_contrast = os.path.join(path, "explain_contrast")
                os.makedirs(out_dir_contrast, exist_ok=True)
                try:
                    print("[Visualization] 生成两类对比的通道级关键 patch 可视化...")
                    fn2 = getattr(visualizer, "run_explain_contrast_viz", None)
                    if callable(fn2):
                        fn2(
                            test_loader,
                            out_dir=out_dir_contrast,
                            max_batches=10,
                            num_channels=int(
                                getattr(self.args, "explain_patch_channels", 4)),
                            topk=int(
                                getattr(self.args, "explain_patch_topk", 6)),
                            candidate_topm=int(
                                getattr(self.args, "explain_contrast_candidate_topm", 12)),
                            layer=-1,
                            min_conf=0.5,
                        )
                except Exception as e:
                    print(f"[WARN] explain_contrast_viz 失败（可忽略）: {e}")

            if is_router_model and bool(getattr(self.args, "importance_eval", True)):
                out_dir_imp = os.path.join(path, "importance_robustness")
                os.makedirs(out_dir_imp, exist_ok=True)
                try:
                    print("[Visualization] 运行重要性鲁棒性实验（ex-ante patch_importance）...")
                    fn4 = getattr(
                        visualizer, "run_importance_robustness_eval", None)
                    if callable(fn4):
                        fn4(
                            data_loader=test_loader,
                            out_dir=out_dir_imp,
                            ratios=None,
                            max_batches=int(
                                getattr(self.args, "importance_eval_max_batches", -1)),
                            use_only_correct=True,
                            min_conf=float(
                                getattr(self.args, "importance_eval_min_conf", 0.5)),
                            num_channels=int(
                                getattr(self.args, "explain_patch_channels", 4)),
                            layer=-1,
                            noise_std_scale=0.5,
                        )
                except Exception as e:
                    print(f"[WARN] importance_robustness_eval 失败（可忽略）: {e}")

            if is_router_model and hasattr(test_loader, "dataset") and hasattr(test_loader.dataset, "get_importance_gt_patch_level"):
                try:
                    out_dir_gt = os.path.join(path, "importance_robustness")
                    fn_gt = getattr(
                        visualizer, "run_importance_gt_correlation", None)
                    if callable(fn_gt):
                        fn_gt(test_loader, out_dir_gt, max_batches=10)
                except Exception as e:
                    print(f"[WARN] importance_gt_correlation 失败（可忽略）: {e}")

            print("[Visualization] 精图可视化完成！")

            if is_router_model:
                try:
                    out_dir_cmp = os.path.join(path, "patch_faithfulness")
                    self._maybe_run_posthoc_compare(
                        visualizer=visualizer,
                        data_loader=test_loader,
                        out_dir=out_dir_cmp,
                    )
                except Exception as e:
                    print(f"[WARN] posthoc_compare failed: {e}")

                try:
                    out_dir_aopc = os.path.join(path, "aopc")
                    self._maybe_run_aopc(
                        visualizer=visualizer,
                        data_loader=test_loader,
                        out_dir=out_dir_aopc,
                    )
                except Exception as e:
                    print(f"[WARN] AOPC failed: {e}")

                try:
                    out_dir_iou = os.path.join(path, "iou")
                    self._maybe_run_iou(
                        visualizer=visualizer,
                        data_loader=test_loader,
                        out_dir=out_dir_iou,
                    )
                except Exception as e:
                    print(f"[WARN] IoU failed: {e}")

            if is_router_model and bool(getattr(self.args, "conf_risk_viz", False)) and hasattr(self, "_risk_conf_means"):
                try:
                    out_dir_risk = os.path.join(path, "conf_risk_viz")
                    conf_means = np.stack(self._risk_conf_means, axis=0) if len(
                        self._risk_conf_means) > 0 else None
                    conf_lows = np.stack(self._risk_conf_lows, axis=0) if len(
                        self._risk_conf_lows) > 0 else None
                    grad_means = np.stack(self._risk_grad_means, axis=0) if len(
                        self._risk_grad_means) > 0 else None
                    self._save_conf_risk_plots(
                        out_dir_risk, conf_means, conf_lows, grad_means)
                    print(
                        f"[Visualization] conf risk viz saved to: {out_dir_risk}")
                except Exception as e:
                    print(f"[WARN] conf risk viz failed: {e}")

            # # 在现有可视化代码后添加
            # if is_router_model and bool(getattr(self.args, "antehoc_proof_viz", True)):
            #     out_dir_proof = os.path.join(path, "antehoc_proof")
            #     os.makedirs(out_dir_proof, exist_ok=True)
            #     try:
            #         print("[Visualization] 生成 Ante-hoc 三属性证明图...")
            #         visualizer.run_antehoc_proof_visualization(
            #             test_loader,
            #             out_dir=out_dir_proof,
            #             max_samples=int(
            #                 getattr(self.args, "antehoc_proof_samples", 4)),
            #             importance_percentile=float(
            #                 getattr(self.args, "antehoc_importance_percentile", 30.0)),
            #             layer=-1,
            #         )
            #     except Exception as e:
            #         import traceback
            #         print(f"[WARN] antehoc_proof_viz 失败: {e}")
            #         traceback.print_exc()

            # # ✅ 新增：专家统计分析
            # if is_router_model and bool(getattr(self.args, "expert_analysis", True)):
            #     from exp.expert_analyzer import ExpertAnalyzer

            #     out_dir_analysis = os.path.join(
            #         path, "expert_quantitative_analysis")
            #     os.makedirs(out_dir_analysis, exist_ok=True)

            #     try:
            #         print("\n" + "="*60)
            #         print("[Visualization] 运行专家量化分析（统计特征）...")
            #         print("="*60)

            #         analyzer = ExpertAnalyzer(
            #             model=model_eval,
            #             device=self.device,
            #             num_groups=getattr(model_eval, "num_groups", 4),
            #             patch_len=int(
            #                 getattr(getattr(model_eval, "patch_embed", None), "patch_len", 16)),
            #             stride=int(
            #                 getattr(getattr(model_eval, "patch_embed", None), "stride", 8)),
            #         )

            #         analyzer.run_full_analysis(
            #             data_loader=test_loader,
            #             out_dir=out_dir_analysis,
            #             max_batches=int(
            #                 getattr(self.args, "expert_analysis_max_batches", 10)),
            #         )

            #     except Exception as e:
            #         print(f"[WARN] expert_analysis 失败（可忽略）: {e}")
            #         import traceback
            #         traceback.print_exc()

        except Exception as e:
            tb = traceback.format_exc()
            print(f"[WARN] 训练后可视化失败: {e}\n{tb}")
            with open(os.path.join(path, "post_analysis_error.log"), "w", encoding="utf-8") as f:
                f.write(tb)

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='TEST')
        if test:
            print('loading model')
            ckpt_path = os.path.join(
                self.args.checkpoints, setting, 'checkpoint.pth')
            self._load_checkpoint_safely(ckpt_path)

        preds, trues = [], []
        self.model.eval()

        autocast_ctx = torch.autocast(
            device_type=self.device.type,
            dtype=self.amp_dtype,
            enabled=self.amp_enabled
        )

        with torch.no_grad(), autocast_ctx:
            for i, (batch_x, label, padding_mask) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = self._normalize_label(label, self.device)

                if getattr(self.args, "warp_test", False):
                    gen = None
                    if getattr(self.args, "warp_seed", None) is not None:
                        gen = torch.Generator(device=self.device).manual_seed(
                            int(self.args.warp_seed))
                    batch_x = piecewise_time_warp(
                        batch_x,
                        level=int(getattr(self.args, "warp_level", 0)),
                        num_segs=int(getattr(self.args, "warp_num_segs", 4)),
                        random_boundaries=bool(
                            getattr(self.args, "warp_random_boundaries", False)),
                        boundary_jitter=float(
                            getattr(self.args, "warp_boundary_jitter", 0.08)),
                        same_warp_across_channels=bool(
                            getattr(self.args, "warp_same_across_channels", True)),
                        generator=gen,
                    )

                outputs = self.model(batch_x, x_mark_enc=padding_mask)

                preds.append(outputs.detach())
                trues.append(label.detach())

        preds = torch.cat(preds, 0)
        trues = torch.cat(trues, 0)
        print('test shape:', preds.shape, trues.shape)

        probs = torch.softmax(preds, dim=1)
        predictions = torch.argmax(probs, dim=1).cpu().numpy()
        trues_np = trues.flatten().cpu().numpy()

        accuracy = cal_accuracy(predictions, trues_np)
        f1 = f1_score(trues_np, predictions, average='weighted')
        precision = precision_score(trues_np, predictions, average='weighted')
        recall = recall_score(trues_np, predictions, average='weighted')

        print('accuracy: {:.4f}'.format(adjust_float(accuracy)))
        print('f1_score: {:.4f}'.format(adjust_float(f1)))
        print('precision: {:.4f}'.format(adjust_float(precision)))
        print('recall: {:.4f}'.format(adjust_float(recall)))

        folder_path = './results/' + setting + '/'
        os.makedirs(folder_path, exist_ok=True)
        file_name = 'result_classification.txt'
        with open(os.path.join(folder_path, file_name), 'a', encoding='utf-8') as f:
            f.write(setting + "  \n")
            f.write(f'accuracy: {adjust_float(accuracy):.4f}\n')
            f.write(f'f1_score: {adjust_float(f1):.4f}\n')
            f.write(f'precision: {adjust_float(precision):.4f}\n')
            f.write(f'recall: {adjust_float(recall):.4f}\n\n')

        if test and bool(getattr(self.args, "importance_eval", True)):
            try:
                path = os.path.join("./checkpoints", setting)
                out_dir_imp = os.path.join(path, "importance_robustness")
                os.makedirs(out_dir_imp, exist_ok=True)

                model_eval = self.model.module if isinstance(
                    self.model, nn.DataParallel) else self.model
                visualizer = self._make_visualizer(model_eval)
                if visualizer is None:
                    return
                print("[Visualization] (test-only) 运行重要性鲁棒性实验...")
                visualizer.run_importance_robustness_eval(
                    data_loader=test_loader,
                    out_dir=out_dir_imp,
                    ratios=None,
                    max_batches=int(
                        getattr(self.args, "importance_eval_max_batches", -1)),
                    use_only_correct=True,
                    min_conf=float(
                        getattr(self.args, "importance_eval_min_conf", 0.5)),
                    noise_std_scale=0.5,
                )
                if hasattr(test_loader.dataset, "get_importance_gt_patch_level"):
                    visualizer.run_importance_gt_correlation(
                        test_loader, out_dir_imp, max_batches=10)
            except Exception as e:
                print(
                    f"[WARN] (test-only) importance_robustness_eval 失败（可忽略）: {e}")

        if bool(getattr(self.args, "posthoc_compare_eval", False)):
            try:
                path = os.path.join(self.args.checkpoints, setting)
                out_dir_cmp = os.path.join(path, "patch_faithfulness")
                model_eval = self.model.module if isinstance(
                    self.model, nn.DataParallel) else self.model
                visualizer = self._make_visualizer(model_eval)
                if visualizer is not None:
                    self._maybe_run_posthoc_compare(
                        visualizer=visualizer,
                        data_loader=test_loader,
                        out_dir=out_dir_cmp,
                    )
            except Exception as e:
                print(f"[WARN] (test-only) posthoc_compare failed: {e}")

        if bool(getattr(self.args, "aopc_eval", False)):
            try:
                path = os.path.join(self.args.checkpoints, setting)
                out_dir_aopc = os.path.join(path, "aopc")
                model_eval = self.model.module if isinstance(
                    self.model, nn.DataParallel) else self.model
                visualizer = self._make_visualizer(model_eval)
                if visualizer is not None:
                    self._maybe_run_aopc(
                        visualizer=visualizer,
                        data_loader=test_loader,
                        out_dir=out_dir_aopc,
                    )
            except Exception as e:
                print(f"[WARN] AOPC failed: {e}")

        if bool(getattr(self.args, "iou_eval", False)):
            try:
                path = os.path.join(self.args.checkpoints, setting)
                out_dir_iou = os.path.join(path, "iou")
                model_eval = self.model.module if isinstance(
                    self.model, nn.DataParallel) else self.model
                visualizer = self._make_visualizer(model_eval)
                if visualizer is not None:
                    self._maybe_run_iou(
                        visualizer=visualizer,
                        data_loader=test_loader,
                        out_dir=out_dir_iou,
                    )
            except Exception as e:
                print(f"[WARN] IoU failed: {e}")

        return


def adjust_float(x: float) -> float:
    return float(np.format_float_positional(x, trim='-'))


# import torch.nn.functional as F
# from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
# from data_provider.data_factory import data_provider
# from exp.exp_basic import Exp_Basic
# from utils.tools import EarlyStopping, cal_accuracy
# import torch
# import torch.nn as nn
# from torch import optim
# import os
# import time
# import warnings
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.metrics import f1_score, precision_score, recall_score
# import traceback
# from tools.timewarp import piecewise_time_warp
# from exp.visualization import ModelVisualizer
# import math

# warnings.filterwarnings('ignore')


# def plot_loss_curves(train_losses, vali_losses, test_losses, out_dir):
#     plt.figure(figsize=(10, 6))
#     epochs = range(1, len(train_losses) + 1)
#     plt.plot(epochs, train_losses, 'b-', label='Train Loss')
#     plt.plot(epochs, vali_losses, 'r-', label='Vali Loss')
#     plt.plot(epochs, test_losses, 'g--', label='Test Loss')
#     plt.title('Training and Validation Loss')
#     plt.xlabel('Epochs')
#     plt.ylabel('Loss')
#     plt.legend()
#     plt.grid(True, alpha=0.3)
#     plt.tight_layout()
#     plt.savefig(os.path.join(out_dir, 'loss_curve.png'), dpi=300)
#     plt.close()
#     print(f"[Visualization] Loss curve saved to {out_dir}/loss_curve.png")


# class Exp_Classification(Exp_Basic):
#     def __init__(self, args):
#         self.training_stats = {
#             'total_params': 0,
#             'avg_memory_mb': 0.0,
#             'memory_samples': [],
#             'inference_times': [],
#             'training_times': []
#         }
#         super(Exp_Classification, self).__init__(args)

#         self.amp_enabled = bool(getattr(self.args, "amp", False)) and (
#             self.device.type == "cuda")
#         self.amp_dtype = torch.float16 if self.amp_enabled else torch.float32
#         self.scaler = torch.cuda.amp.GradScaler(
#             enabled=self.amp_enabled and self.amp_dtype == torch.float16)

#         # 记录Loss
#         self.train_loss_recorder = []
#         self.vali_loss_recorder = []
#         self.test_loss_recorder = []

#     def _build_model(self):
#         torch.manual_seed(self.args.seed)
#         if torch.cuda.is_available():
#             torch.cuda.manual_seed(self.args.seed)

#         from data_provider.uea import collate_fn
#         from torch.utils.data import DataLoader

#         train_data, _ = data_provider(self.args, 'TRAIN')
#         test_data, _ = data_provider(self.args, 'TEST')
#         vali_data, _ = data_provider(self.args, 'TEST')

#         dataset_max_seq_len = max(
#             train_data.max_seq_len, test_data.max_seq_len, vali_data.max_seq_len)
#         if hasattr(self.args, 'seq_len') and self.args.seq_len is not None:
#             if self.args.seq_len < dataset_max_seq_len:
#                 print(
#                     f"警告: 显式设置的seq_len={self.args.seq_len}小于数据集最大长度={dataset_max_seq_len}，使用数据集最大值避免截断")
#                 self.args.seq_len = dataset_max_seq_len
#         else:
#             self.args.seq_len = dataset_max_seq_len

#         train_loader = DataLoader(
#             train_data,
#             batch_size=self.args.batch_size,
#             shuffle=True,
#             num_workers=self.args.num_workers,
#             drop_last=False,
#             collate_fn=lambda x: collate_fn(x, max_len=self.args.seq_len)
#         )
#         test_loader = DataLoader(
#             test_data,
#             batch_size=self.args.batch_size,
#             shuffle=False,
#             num_workers=self.args.num_workers,
#             drop_last=False,
#             collate_fn=lambda x: collate_fn(x, max_len=self.args.seq_len)
#         )

#         self.args.pred_len = 0
#         self.args.enc_in = train_data.feature_df.shape[1]
#         self.args.num_class = len(train_data.class_names)
#         self.args.class_names = train_data.class_names

#         model = self.model_dict[self.args.model](self.args).float()

#         total = sum(p.numel() for p in model.parameters())
#         trainable = sum(p.numel()
#                         for p in model.parameters() if p.requires_grad)
#         frozen = total - trainable
#         self.training_stats['total_params'] = total
#         self.training_stats['trainable_params'] = trainable
#         self.training_stats['frozen_params'] = frozen

#         print(f"=== Model Params ===")
#         print(f"Total     : {total:,}")
#         print(f"Trainable : {trainable:,}")
#         print(f"Frozen    : {frozen:,}")

#         if getattr(self.args, "use_multi_gpu", False) and getattr(self.args, "use_gpu", False):
#             model = nn.DataParallel(model, device_ids=getattr(
#                 self.args, "device_ids", None))

#         self.enc_in = self.args.enc_in

#         self.fs = None
#         for ds in [train_data, test_data, vali_data]:
#             if hasattr(ds, 'fs'):
#                 self.fs = float(getattr(ds, 'fs'))
#                 break
#             if hasattr(ds, 'sampling_rate'):
#                 self.fs = float(getattr(ds, 'sampling_rate'))
#                 break
#         if self.fs is None:
#             self.fs = 1.0
#             print(
#                 "[analysis] fs not provided; using normalized frequency (cycles/sample, Nyquist=0.5)")
#         else:
#             print(f"[analysis] sampling rate fs = {self.fs} Hz")

#         return model

#     def _get_data(self, flag):
#         data_set, data_loader = data_provider(self.args, flag)
#         return data_set, data_loader

#     def _select_optimizer(self):
#         model_optim = optim.Adam(
#             self.model.parameters(),
#             lr=self.args.learning_rate,
#             betas=(0.9, getattr(self.args, 'beta2', 0.99))
#         )
#         t0 = int(getattr(self.args, "cawr_t0", 10))
#         tm = int(getattr(self.args, "cawr_tmult", 2))
#         eta_min = float(getattr(self.args, "min_lr",
#                         self.args.learning_rate * 1e-3))
#         self.scheduler = CosineAnnealingWarmRestarts(
#             model_optim, T_0=max(1, t0), T_mult=max(1, tm), eta_min=eta_min
#         )
#         return model_optim

#     def _select_criterion(self):
#         import loss
#         import importlib
#         importlib.reload(loss)
#         # 使用新的统一 Loss 类
#         return loss.NewModelLoss(self.args)

#     def _load_checkpoint_safely(self, ckpt_path: str):
#         if not os.path.exists(ckpt_path):
#             raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

#         ckpt = torch.load(ckpt_path, map_location=self.device)

#         if isinstance(ckpt, dict):
#             if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
#                 state = ckpt["state_dict"]
#             elif "model" in ckpt and isinstance(ckpt["model"], dict):
#                 state = ckpt["model"]
#             else:
#                 tensor_dict = {
#                     k: v for k, v in ckpt.items() if isinstance(v, torch.Tensor)}
#                 state = tensor_dict if len(tensor_dict) > 0 else ckpt
#         else:
#             state = ckpt

#         model_is_dp = isinstance(self.model, nn.DataParallel)
#         keys = list(state.keys())
#         has_module_prefix = (len(keys) > 0 and keys[0].startswith("module."))

#         if model_is_dp and not has_module_prefix:
#             state = {f"module.{k}": v for k, v in state.items()}
#         if (not model_is_dp) and has_module_prefix:
#             state = {k.replace("module.", "", 1): v for k, v in state.items()}

#         missing, unexpected = self.model.load_state_dict(state, strict=False)
#         if missing or unexpected:
#             print("[WARN] load_state_dict strict=False:")
#             if missing:
#                 print("  missing keys:", missing[:20], "..." if len(
#                     missing) > 20 else "")
#             if unexpected:
#                 print("  unexpected keys:", unexpected[:20], "..." if len(
#                     unexpected) > 20 else "")

#         print(f"[OK] Loaded checkpoint: {ckpt_path}")

#     def visualize_from_checkpoint(self, ckpt_path: str, max_batches: int = 4, topk: int = 2):
#         _, test_loader = self._get_data(flag='TEST')
#         self._load_checkpoint_safely(ckpt_path)
#         self.model.eval()

#         setting = os.path.basename(os.path.dirname(ckpt_path))
#         out_root = os.path.join(self.args.checkpoints, setting, "viz_only")
#         os.makedirs(out_root, exist_ok=True)

#         model_eval = self.model.module if isinstance(
#             self.model, nn.DataParallel) else self.model
#         visualizer = ModelVisualizer(
#             model=model_eval,
#             device=self.device,
#             enc_in=self.enc_in,
#             fs=self.fs,
#             class_names=getattr(self.args, "class_names", None),
#         )

#         out_dir_amp = os.path.join(out_root, "paper_ampcase")
#         os.makedirs(out_dir_amp, exist_ok=True)
#         if hasattr(model_eval, "inspect_aux"):
#             print("[Visualization] 生成 paper_story_only（频率-语义对齐）...")
#             visualizer.run_paper_story_only(
#                 test_loader,
#                 out_dir=out_dir_amp,
#                 max_batches=8,
#                 top_channels=4,
#                 max_stories=3,
#             )
#         else:
#             print("[Visualization] 跳过 paper_story_only：当前模型未实现 inspect_aux。")

#         out_dir_feat = os.path.join(out_root, "feature_viz")
#         os.makedirs(out_dir_feat, exist_ok=True)
#         print("[Visualization] feature viz (t-SNE)...")
#         visualizer.run_feature_viz(
#             test_loader,
#             out_dir=out_dir_feat,
#             take="prepool",
#             hook_module=model_eval.norm
#         )

#         out_dir_vec = os.path.join(out_root, "vectorosc_aux")
#         os.makedirs(out_dir_vec, exist_ok=True)
#         print("[Visualization] VectorOsc aux viz (freq/amp/waves/harm_weights)...")
#         try:
#             from utils.viz_utils import plot_vectorosc_aux
#         except Exception:
#             plot_vectorosc_aux = None

#         if plot_vectorosc_aux is not None:
#             with torch.no_grad():
#                 for bi, batch in enumerate(test_loader):
#                     if bi >= int(max_batches):
#                         break

#                     batch_x, label, padding_mask = batch[0], batch[1], batch[2] if len(
#                         batch) > 2 else None
#                     batch_x = batch_x.float().to(self.device)
#                     padding_mask = padding_mask.float().to(
#                         self.device) if padding_mask is not None else None
#                     label = self._normalize_label(label, self.device)
#                     if label.numel() == 0:
#                         continue

#                     aux_list = None
#                     outputs = None
#                     if hasattr(model_eval, 'forward_with_recon'):
#                         outputs, _, aux_list = model_eval.forward_with_recon(
#                             batch_x, x_mark_enc=padding_mask
#                         )
#                     else:
#                         outputs = self.model(
#                             batch_x, x_mark_enc=padding_mask, return_aux=True)
#                         # 兼容 (logits, aux_dict) 返回
#                         if isinstance(outputs, (tuple, list)) and len(outputs) >= 1:
#                             outputs = outputs[0]

#                     if outputs is None:
#                         continue
#                     if outputs.dim() == 3:
#                         outputs = outputs.mean(dim=1)
#                     pred = torch.argmax(outputs, dim=-1).view(-1)

#                     y0 = int(label.view(-1)[0].item())
#                     p0 = int(pred.view(-1)[0].item())
#                     tag = f"b{bi}_y{y0}_pred{p0}"

#                     if isinstance(aux_list, (list, tuple)) and len(aux_list) > 0:
#                         plot_vectorosc_aux(
#                             aux_list=aux_list,
#                             out_dir=out_dir_vec,
#                             sample_idx=0,
#                             layers=[-1],
#                             topk_heads=int(topk),
#                             tag_prefix=tag,
#                         )

#         print(f"[OK] Visualization saved to: {out_root}")

#     def _save_training_stats(self, setting):
#         path = os.path.join(self.args.checkpoints, setting)
#         os.makedirs(path, exist_ok=True)

#         stats_file = os.path.join(path, "training_stats.txt")

#         if len(self.training_stats['memory_samples']) > 0:
#             avg_memory = np.mean(self.training_stats['memory_samples'])
#             self.training_stats['avg_memory_mb'] = avg_memory
#         else:
#             avg_memory = 0.0

#         avg_inference_time = np.mean(self.training_stats['inference_times']) if len(
#             self.training_stats['inference_times']) > 0 else 0.0
#         avg_training_time = np.mean(self.training_stats['training_times']) if len(
#             self.training_stats['training_times']) > 0 else 0.0

#         with open(stats_file, 'w', encoding='utf-8') as f:
#             f.write("=== Training Statistics ===\n")
#             f.write(
#                 f"Total parameters: {self.training_stats['total_params']:,}\n")
#             f.write(f"Average GPU memory usage: {avg_memory:.2f} MB\n")
#             f.write(
#                 f"Average inference time per iter: {avg_inference_time*1000:.4f} ms\n")
#             f.write(
#                 f"Average training time per iter: {avg_training_time*1000:.4f} ms\n")

#         print("=== Training Statistics ===")
#         print(f"Total parameters: {self.training_stats['total_params']:,}")
#         print(f"Average GPU memory usage: {avg_memory:.2f} MB")
#         print(
#             f"Average inference time per iter: {avg_inference_time*1000:.4f} ms")
#         print(
#             f"Average training time per iter: {avg_training_time*1000:.4f} ms")
#         print(f"Training stats saved to: {stats_file}")

#         if len(self.train_loss_recorder) > 0:
#             plot_loss_curves(
#                 self.train_loss_recorder,
#                 self.vali_loss_recorder,
#                 self.test_loss_recorder,
#                 path
#             )

#     def _update_memory_stats(self):
#         if torch.cuda.is_available():
#             current_memory = torch.cuda.memory_allocated() / 1024**2
#             self.training_stats['memory_samples'].append(current_memory)

#     @torch.no_grad()
#     def _collect_confidence_stats(self, loader, max_batches: int, conf_low_th: float):
#         model_eval = self.model.module if isinstance(
#             self.model, nn.DataParallel) else self.model
#         model_eval.eval()

#         conf_list = []
#         for bi, batch in enumerate(loader):
#             if bi >= int(max_batches):
#                 break
#             batch_x, _, padding_mask = batch[0], batch[1], batch[2] if len(
#                 batch) > 2 else None
#             batch_x = batch_x.float().to(self.device)
#             padding_mask = padding_mask.float().to(
#                 self.device) if padding_mask is not None else None

#             ret = model_eval(batch_x, x_mark_enc=padding_mask, return_aux=True)
#             if not (isinstance(ret, (tuple, list)) and len(ret) == 2 and isinstance(ret[1], dict)):
#                 continue
#             aux = ret[1]

#             # 兼容新模型：直接在 aux 里找 confidences
#             conf = aux.get("confidences", None)

#             # 如果没有，尝试从 layers 里找 (旧模型兼容)
#             if conf is None:
#                 layers = aux.get("layers", None)
#                 if isinstance(layers, list) and len(layers) > 0:
#                     last = layers[-1]
#                     if isinstance(last, dict):
#                         conf = last.get("confidences", None)

#             if conf is None or (not torch.is_tensor(conf)):
#                 continue

#             # 统一维度处理：如果是 [B, Lp, K, 1] 转为 [B, Lp, K]
#             if conf.dim() == 4:
#                 conf = conf.squeeze(-1)

#             # 如果是 [B, Lp, K]，求平均得到 [B, K]
#             if conf.dim() == 3:
#                 conf = conf.mean(dim=1)

#             conf_list.append(conf.detach().float().cpu())

#         if len(conf_list) == 0:
#             return None, None

#         conf_all = torch.cat(conf_list, dim=0)  # [Btot,K]
#         conf_mean = conf_all.mean(dim=0)        # [K]
#         conf_low_rate = (conf_all < float(conf_low_th)
#                          ).float().mean(dim=0)  # [K]
#         return conf_mean.numpy(), conf_low_rate.numpy()

#     @staticmethod
#     def _extract_expert_modules(model_eval):
#         # 兼容新模型结构
#         if hasattr(model_eval, "review_block") and hasattr(model_eval.review_block, "experts"):
#             return list(model_eval.review_block.experts)
#         # 兼容旧模型结构
#         if hasattr(model_eval, "shared_experts"):
#             return list(model_eval.shared_experts)
#         return []

#     def _collect_grad_norms(self, model_eval):
#         experts = self._extract_expert_modules(model_eval)
#         if len(experts) == 0:
#             return None
#         vals = []
#         for ex in experts:
#             s = 0.0
#             for p in ex.parameters():
#                 if p.grad is None:
#                     continue
#                 g = p.grad.detach()
#                 s += float(torch.norm(g, p=2).item())
#             vals.append(s)
#         return np.array(vals, dtype=np.float32)

#     @staticmethod
#     def _save_conf_risk_plots(out_dir: str, conf_means, conf_lows, grad_means):
#         os.makedirs(out_dir, exist_ok=True)

#         def _heatmap(arr, title, fname, vmin=None, vmax=None):
#             plt.figure(figsize=(10, 3.6))
#             plt.imshow(arr, aspect="auto", interpolation="nearest",
#                        vmin=vmin, vmax=vmax, cmap="viridis")
#             plt.colorbar()
#             plt.title(title)
#             plt.xlabel("expert (K)")
#             plt.ylabel("epoch")
#             plt.tight_layout()
#             plt.savefig(os.path.join(out_dir, fname), dpi=300)
#             plt.close()

#         if conf_means is not None:
#             _heatmap(conf_means, "Confidence mean (last layer)",
#                      "conf_mean_heatmap.png", vmin=0.0, vmax=1.0)
#         if conf_lows is not None:
#             _heatmap(conf_lows, "Low-confidence rate (conf < th)",
#                      "conf_lowrate_heatmap.png", vmin=0.0, vmax=1.0)
#         if grad_means is not None:
#             _heatmap(np.log10(grad_means + 1e-12),
#                      "log10(grad norm) per expert", "grad_norm_log10_heatmap.png")

#         try:
#             plt.figure(figsize=(8, 4))
#             if conf_means is not None:
#                 plt.plot(conf_means.mean(axis=1), label="mean(conf)")
#             if conf_lows is not None:
#                 plt.plot(conf_lows.mean(axis=1), label="mean(low-rate)")
#             plt.title("Expert silence risk trends")
#             plt.xlabel("epoch")
#             plt.ylabel("value")
#             plt.grid(True, alpha=0.25)
#             plt.legend()
#             plt.tight_layout()
#             plt.savefig(os.path.join(out_dir, "risk_trends.png"), dpi=300)
#             plt.close()
#         except Exception:
#             pass

#     @staticmethod
#     def _normalize_label(label: torch.Tensor, device: torch.device) -> torch.Tensor:
#         label = label.to(device).long()
#         if label.dim() > 1:
#             label = label.view(label.size(0), -1)[:, 0]
#         return label.view(-1)

#     def vali(self, vali_data, vali_loader, criterion):
#         self.model.eval()
#         total_loss = []
#         total_correct = 0
#         total_samples = 0

#         autocast_ctx = torch.autocast(
#             device_type=self.device.type,
#             dtype=self.amp_dtype,
#             enabled=self.amp_enabled
#         )

#         with torch.no_grad(), autocast_ctx:
#             for i, (batch_x, label, padding_mask) in enumerate(vali_loader):
#                 batch_x = batch_x.float().to(self.device)
#                 padding_mask = padding_mask.float().to(self.device)
#                 label = self._normalize_label(label, self.device)

#                 if label.numel() == 0:
#                     continue

#                 outputs = self.model(
#                     batch_x, x_mark_enc=padding_mask, return_aux=False)

#                 if outputs.dim() == 3:
#                     outputs = outputs.mean(dim=1)

#                 # 验证集通常只看分类损失，不传 model/batch_x 计算复杂损失
#                 loss, _ = criterion(outputs, label)
#                 probs = torch.softmax(outputs, dim=1)
#                 predictions = torch.argmax(probs, dim=1).view(-1)
#                 correct = (predictions == label).sum().item()

#                 total_loss.append(loss.item())
#                 total_correct += correct
#                 total_samples += label.size(0)

#         if len(total_loss) == 0 or total_samples == 0:
#             return 0.0, 0.0

#         avg_loss = float(np.average(total_loss))
#         accuracy = float(total_correct) / float(total_samples)

#         self.model.train()
#         return avg_loss, accuracy

#     def train(self, setting):
#         if torch.cuda.is_available():
#             torch.cuda.reset_peak_memory_stats()

#         train_data, train_loader = self._get_data(flag='TRAIN')
#         vali_data, vali_loader = self._get_data(flag='TEST')
#         test_data, test_loader = self._get_data(flag='TEST')

#         path = os.path.join(self.args.checkpoints, setting)
#         os.makedirs(path, exist_ok=True)

#         best_test_acc = -float("inf")
#         best_test_path = os.path.join(path, "best_test_acc.pth")

#         time_now = time.time()
#         train_steps = len(train_loader)
#         early_stopping = EarlyStopping(
#             patience=self.args.patience,
#             verbose=True,
#             mode='max'
#         )

#         model_optim = self._select_optimizer()
#         criterion = self._select_criterion()

#         for epoch in range(self.args.train_epochs):
#             epoch_start_time = time.time()
#             iter_count = 0
#             train_loss = []
#             loss_terms_acc = {}

#             self.model.train()

#             grad_every = int(getattr(self.args, "grad_risk_every", 0))
#             if grad_every <= 0:
#                 grad_every = max(1, train_steps // 10)
#             grad_acc = None
#             grad_n = 0

#             for i, batch in enumerate(train_loader):
#                 if len(batch) == 4:
#                     batch_x, label, padding_mask, importance_gt_batch = batch
#                 else:
#                     batch_x, label, padding_mask = batch
#                     importance_gt_batch = None

#                 iter_count += 1
#                 iter_start_time = time.time()
#                 model_optim.zero_grad(set_to_none=True)

#                 batch_x = batch_x.float().to(self.device)
#                 padding_mask = padding_mask.float().to(self.device)
#                 label = self._normalize_label(label, self.device)
#                 if importance_gt_batch is not None:
#                     importance_gt_batch = importance_gt_batch.to(self.device)

#                 inference_start = time.time()
#                 with torch.autocast(
#                     device_type=self.device.type,
#                     dtype=self.amp_dtype,
#                     enabled=self.amp_enabled
#                 ):
#                     model_eval = self.model.module if isinstance(
#                         self.model, nn.DataParallel) else self.model

#                     if hasattr(model_eval, 'forward_with_recon'):
#                         outputs, _, _ = model_eval.forward_with_recon(
#                             batch_x, x_mark_enc=padding_mask
#                         )
#                     else:
#                         outputs = self.model(batch_x, x_mark_enc=padding_mask)

#                     # 统一调用 Loss 计算
#                     loss, loss_dict = criterion(
#                         outputs=outputs,
#                         targets=label,
#                         batch_x=batch_x,
#                         padding_mask=padding_mask,
#                         importance_gt=importance_gt_batch,
#                         model=self.model
#                     )

#                 inference_time = time.time() - inference_start
#                 self.training_stats['inference_times'].append(inference_time)

#                 for k, v in loss_dict.items():
#                     loss_terms_acc.setdefault(k, []).append(v)

#                 if self.scaler.is_enabled():
#                     self.scaler.scale(loss).backward()
#                     self.scaler.unscale_(model_optim)
#                     nn.utils.clip_grad_norm_(
#                         self.model.parameters(), max_norm=4.0)

#                     if bool(getattr(self.args, "conf_risk_viz", False)) and ((i % grad_every) == 0):
#                         g = self._collect_grad_norms(model_eval)
#                         if g is not None:
#                             grad_acc = g if grad_acc is None else (
#                                 grad_acc + g)
#                             grad_n += 1
#                     self.scaler.step(model_optim)
#                     self.scaler.update()
#                 else:
#                     loss.backward()
#                     nn.utils.clip_grad_norm_(
#                         self.model.parameters(), max_norm=4.0)
#                     if bool(getattr(self.args, "conf_risk_viz", False)) and ((i % grad_every) == 0):
#                         g = self._collect_grad_norms(model_eval)
#                         if g is not None:
#                             grad_acc = g if grad_acc is None else (
#                                 grad_acc + g)
#                             grad_n += 1
#                     model_optim.step()

#                 training_time = time.time() - iter_start_time
#                 self.training_stats['training_times'].append(training_time)
#                 self._update_memory_stats()

#                 try:
#                     self.scheduler.step(epoch + (i + 1) / train_steps)
#                 except Exception:
#                     pass

#                 train_loss.append(loss.item())

#                 if (i + 1) % 100 == 0:
#                     terms_str = ", ".join(
#                         [f"{k}:{np.mean(v):.6f}" for k, v in loss_terms_acc.items() if len(v) > 0])
#                     print(
#                         f"\titers: {i+1}, epoch: {epoch+1} | loss: {loss.item():.7f} | {terms_str}")
#                     speed = (time.time() - time_now) / iter_count
#                     left_time = speed * \
#                         ((self.args.train_epochs - epoch) * train_steps - i)
#                     print(
#                         f"\tspeed: {speed:.4f}s/iter; left time: {left_time:.4f}s")
#                     iter_count = 0
#                     time_now = time.time()

#             epoch_train_loss = float(np.average(train_loss)) if len(
#                 train_loss) > 0 else 0.0
#             self.train_loss_recorder.append(epoch_train_loss)

#             epoch_duration = time.time() - epoch_start_time
#             print("Epoch: {} cost time: {:.2f}s".format(
#                 epoch + 1, epoch_duration))

#             vali_loss, val_accuracy = self.vali(
#                 vali_data, vali_loader, criterion)
#             test_loss, test_accuracy = self.vali(
#                 test_data, test_loader, criterion)

#             self.vali_loss_recorder.append(vali_loss)
#             self.test_loss_recorder.append(test_loss)

#             print(
#                 "Epoch: {0}, Steps: {1} | Train Loss: {2:.3f} Vali Loss: {3:.3f} "
#                 "Vali Acc: {4:.3f} Test Loss: {5:.3f} Test Acc: {6:.3f}"
#                 .format(epoch + 1, train_steps, epoch_train_loss, vali_loss, val_accuracy, test_loss, test_accuracy)
#             )

#             try:
#                 if test_accuracy > best_test_acc:
#                     best_test_acc = float(test_accuracy)
#                     torch.save(self.model.state_dict(), best_test_path)
#             except Exception as e:
#                 print(f"[WARN] saving best_test failed: {e}")

#             early_stopping(val_accuracy, self.model, path)
#             if early_stopping.early_stop:
#                 print("Early stopping")
#                 break

#             if bool(getattr(self.args, "enable_viz", True)) and bool(getattr(self.args, "conf_risk_viz", False)):
#                 try:
#                     if not hasattr(self, "_risk_conf_means"):
#                         self._risk_conf_means = []
#                         self._risk_conf_lows = []
#                         self._risk_grad_means = []
#                     conf_mean, conf_low = self._collect_confidence_stats(
#                         train_loader,
#                         max_batches=int(
#                             getattr(self.args, "conf_risk_max_batches", 4)),
#                         conf_low_th=float(
#                             getattr(self.args, "conf_low_th", 0.1)),
#                     )
#                     if conf_mean is not None:
#                         self._risk_conf_means.append(conf_mean)
#                         self._risk_conf_lows.append(conf_low)
#                     if grad_acc is not None and grad_n > 0:
#                         self._risk_grad_means.append(
#                             (grad_acc / float(grad_n)))
#                 except Exception:
#                     pass

#         self._update_memory_stats()
#         self._save_training_stats(setting)

#         best_candidates = [
#             os.path.join(path, "best_test_acc.pth"),
#             os.path.join(path, "checkpoint.pth"),
#         ]
#         loaded = False
#         for ckpt in best_candidates:
#             if os.path.exists(ckpt):
#                 try:
#                     self._load_checkpoint_safely(ckpt)
#                     loaded = True
#                     break
#                 except Exception as e:
#                     print(f"[WARN] failed to load {ckpt}: {e}")
#         if not loaded:
#             print("[WARN] No checkpoint loaded; using current in-memory weights.")

#         try:
#             model_eval = self.model.module if isinstance(
#                 self.model, nn.DataParallel) else self.model
#             visualizer = ModelVisualizer(
#                 model=model_eval,
#                 device=self.device,
#                 enc_in=self.enc_in,
#                 fs=self.fs,
#                 class_names=getattr(self.args, "class_names", None),
#             )

#             # ✅ 关键修复：增加对 review_block 的检测
#             is_router_model = bool(
#                 hasattr(model_eval, "use_router_system")
#                 or hasattr(model_eval, "shared_experts")
#                 or hasattr(model_eval, "review_block")  # <--- 新增
#             )

#             if not bool(getattr(self.args, "enable_viz", True)):
#                 if is_router_model and bool(getattr(self.args, "importance_eval", True)):
#                     out_dir_imp = os.path.join(path, "importance_robustness")
#                     os.makedirs(out_dir_imp, exist_ok=True)
#                     try:
#                         print(
#                             "[Visualization] enable_viz=False，仅运行重要性鲁棒性评估（MoRF/LeRF）...")
#                         fn4 = getattr(
#                             visualizer, "run_importance_robustness_eval", None)
#                         if callable(fn4):
#                             fn4(
#                                 data_loader=test_loader,
#                                 out_dir=out_dir_imp,
#                                 ratios=None,
#                                 max_batches=int(
#                                     getattr(self.args, "importance_eval_max_batches", -1)),
#                                 use_only_correct=True,
#                                 min_conf=float(
#                                     getattr(self.args, "importance_eval_min_conf", 0.5)),
#                                 num_channels=int(
#                                     getattr(self.args, "explain_patch_channels", 4)),
#                                 layer=-1,
#                                 noise_std_scale=0.5,
#                             )
#                         fn_gt = getattr(
#                             visualizer, "run_importance_gt_correlation", None)
#                         if callable(fn_gt) and hasattr(test_loader, "dataset") and hasattr(test_loader.dataset, "get_importance_gt_patch_level"):
#                             fn_gt(test_loader, out_dir_imp, max_batches=10)
#                     except Exception as e:
#                         print(f"[WARN] importance_eval(light) 失败（可忽略）: {e}")
#                 else:
#                     print("[Visualization] enable_viz=False，跳过训练后重可视化。")
#                 return self.model

#             out_dir_amp = os.path.join(path, "paper_ampcase")
#             os.makedirs(out_dir_amp, exist_ok=True)
#             print("[Visualization] 生成 paper_story_only（仅证据级图）...")
#             visualizer.run_paper_story_only(
#                 test_loader,
#                 out_dir=out_dir_amp,
#                 max_batches=8,
#                 top_channels=4,
#                 max_stories=2,
#             )

#             out_dir_feat = os.path.join(path, "feature_viz")
#             os.makedirs(out_dir_feat, exist_ok=True)
#             print("[Visualization] 生成 t-SNE...")
#             hook_module = getattr(model_eval, "norm", None)
#             if hook_module is None:
#                 hook_module = getattr(
#                     model_eval, "layer_norm", None) or model_eval
#             visualizer.run_feature_viz(
#                 test_loader,
#                 out_dir=out_dir_feat,
#                 take="prepool",
#                 hook_module=hook_module
#             )

#             if is_router_model:
#                 out_dir_attn = os.path.join(path, "router_attn_viz")
#                 os.makedirs(out_dir_attn, exist_ok=True)
#                 try:
#                     print("[Visualization] 生成 router anchor-attention 可视化...")
#                     visualizer.run_router_attn_viz(
#                         test_loader,
#                         out_dir=out_dir_attn,
#                         max_batches=2,
#                         sample_idx=0,
#                         layers=[-1],
#                     )
#                 except Exception as e:
#                     print(f"[WARN] router_attn_viz 失败（可忽略）: {e}")

#             if is_router_model and bool(getattr(self.args, "anchor_tsne_viz", True)):
#                 out_dir_tsne = os.path.join(path, "anchor_tsne")
#                 os.makedirs(out_dir_tsne, exist_ok=True)
#                 try:
#                     print("[Visualization] 生成各层 anchors t-SNE 可视化...")
#                     visualizer.run_anchor_tsne_viz(
#                         test_loader,
#                         out_dir=out_dir_tsne,
#                         max_batches=int(
#                             getattr(self.args, "anchor_tsne_max_batches", 3)),
#                         perplexity=int(
#                             getattr(self.args, "anchor_tsne_perplexity", 30)),
#                         max_points=int(
#                             getattr(self.args, "anchor_tsne_max_points", 2000)),
#                     )
#                 except Exception as e:
#                     print(f"[WARN] anchor_tsne_viz 失败（可忽略）: {e}")

#             if is_router_model and bool(getattr(self.args, "explain_patch_viz", True)):
#                 out_dir_explain = os.path.join(path, "explain_patches")
#                 os.makedirs(out_dir_explain, exist_ok=True)
#                 try:
#                     print("[Visualization] 生成最终决策高置信度 patch 可视化...")
#                     fn = getattr(visualizer, "run_explain_patch_viz", None)
#                     if callable(fn):
#                         fn(
#                             test_loader,
#                             out_dir=out_dir_explain,
#                             max_batches=int(
#                                 getattr(self.args, "explain_patch_max_batches", 2)),
#                             max_samples=int(
#                                 getattr(self.args, "explain_patch_max_samples", 4)),
#                             topk=int(
#                                 getattr(self.args, "explain_patch_topk", 6)),
#                             num_channels=int(
#                                 getattr(self.args, "explain_patch_channels", 4)),
#                             layer=-1,
#                         )
#                 except Exception as e:
#                     print(f"[WARN] explain_patch_viz 失败（可忽略）: {e}")

#             if is_router_model and bool(getattr(self.args, "explain_contrast_viz", True)):
#                 out_dir_contrast = os.path.join(path, "explain_contrast")
#                 os.makedirs(out_dir_contrast, exist_ok=True)
#                 try:
#                     print("[Visualization] 生成两类对比的通道级关键 patch 可视化...")
#                     fn2 = getattr(visualizer, "run_explain_contrast_viz", None)
#                     if callable(fn2):
#                         fn2(
#                             test_loader,
#                             out_dir=out_dir_contrast,
#                             max_batches=10,
#                             num_channels=int(
#                                 getattr(self.args, "explain_patch_channels", 4)),
#                             topk=int(
#                                 getattr(self.args, "explain_patch_topk", 6)),
#                             candidate_topm=int(
#                                 getattr(self.args, "explain_contrast_candidate_topm", 12)),
#                             layer=-1,
#                             min_conf=0.5,
#                         )
#                 except Exception as e:
#                     print(f"[WARN] explain_contrast_viz 失败（可忽略）: {e}")

#             if is_router_model and bool(getattr(self.args, "importance_eval", True)):
#                 out_dir_imp = os.path.join(path, "importance_robustness")
#                 os.makedirs(out_dir_imp, exist_ok=True)
#                 try:
#                     print("[Visualization] 运行重要性鲁棒性实验（ex-ante patch_importance）...")
#                     fn4 = getattr(
#                         visualizer, "run_importance_robustness_eval", None)
#                     if callable(fn4):
#                         fn4(
#                             data_loader=test_loader,
#                             out_dir=out_dir_imp,
#                             ratios=None,
#                             max_batches=int(
#                                 getattr(self.args, "importance_eval_max_batches", -1)),
#                             use_only_correct=True,
#                             min_conf=float(
#                                 getattr(self.args, "importance_eval_min_conf", 0.5)),
#                             num_channels=int(
#                                 getattr(self.args, "explain_patch_channels", 4)),
#                             layer=-1,
#                             noise_std_scale=0.5,
#                         )
#                 except Exception as e:
#                     print(f"[WARN] importance_robustness_eval 失败（可忽略）: {e}")

#             if is_router_model and hasattr(test_loader, "dataset") and hasattr(test_loader.dataset, "get_importance_gt_patch_level"):
#                 try:
#                     out_dir_gt = os.path.join(path, "importance_robustness")
#                     fn_gt = getattr(
#                         visualizer, "run_importance_gt_correlation", None)
#                     if callable(fn_gt):
#                         fn_gt(test_loader, out_dir_gt, max_batches=10)
#                 except Exception as e:
#                     print(f"[WARN] importance_gt_correlation 失败（可忽略）: {e}")

#             print("[Visualization] 精图可视化完成！")

#             if is_router_model and bool(getattr(self.args, "conf_risk_viz", False)) and hasattr(self, "_risk_conf_means"):
#                 try:
#                     out_dir_risk = os.path.join(path, "conf_risk_viz")
#                     conf_means = np.stack(self._risk_conf_means, axis=0) if len(
#                         self._risk_conf_means) > 0 else None
#                     conf_lows = np.stack(self._risk_conf_lows, axis=0) if len(
#                         self._risk_conf_lows) > 0 else None
#                     grad_means = np.stack(self._risk_grad_means, axis=0) if len(
#                         self._risk_grad_means) > 0 else None
#                     self._save_conf_risk_plots(
#                         out_dir_risk, conf_means, conf_lows, grad_means)
#                     print(
#                         f"[Visualization] conf risk viz saved to: {out_dir_risk}")
#                 except Exception as e:
#                     print(f"[WARN] conf risk viz failed: {e}")

#         except Exception as e:
#             tb = traceback.format_exc()
#             print(f"[WARN] 训练后可视化失败: {e}\n{tb}")
#             with open(os.path.join(path, "post_analysis_error.log"), "w", encoding="utf-8") as f:
#                 f.write(tb)

#         return self.model

#     def test(self, setting, test=0):
#         test_data, test_loader = self._get_data(flag='TEST')
#         if test:
#             print('loading model')
#             ckpt_path = os.path.join(
#                 './checkpoints/' + setting, 'checkpoint.pth')
#             self._load_checkpoint_safely(ckpt_path)

#         preds, trues = [], []
#         self.model.eval()

#         autocast_ctx = torch.autocast(
#             device_type=self.device.type,
#             dtype=self.amp_dtype,
#             enabled=self.amp_enabled
#         )

#         with torch.no_grad(), autocast_ctx:
#             for i, (batch_x, label, padding_mask) in enumerate(test_loader):
#                 batch_x = batch_x.float().to(self.device)
#                 padding_mask = padding_mask.float().to(self.device)
#                 label = self._normalize_label(label, self.device)

#                 if getattr(self.args, "warp_test", False):
#                     gen = None
#                     if getattr(self.args, "warp_seed", None) is not None:
#                         gen = torch.Generator(device=self.device).manual_seed(
#                             int(self.args.warp_seed))
#                     batch_x = piecewise_time_warp(
#                         batch_x,
#                         level=int(getattr(self.args, "warp_level", 0)),
#                         num_segs=int(getattr(self.args, "warp_num_segs", 4)),
#                         random_boundaries=bool(
#                             getattr(self.args, "warp_random_boundaries", False)),
#                         boundary_jitter=float(
#                             getattr(self.args, "warp_boundary_jitter", 0.08)),
#                         same_warp_across_channels=bool(
#                             getattr(self.args, "warp_same_across_channels", True)),
#                         generator=gen,
#                     )

#                 outputs = self.model(batch_x, x_mark_enc=padding_mask)

#                 preds.append(outputs.detach())
#                 trues.append(label.detach())

#         preds = torch.cat(preds, 0)
#         trues = torch.cat(trues, 0)
#         print('test shape:', preds.shape, trues.shape)

#         probs = torch.softmax(preds, dim=1)
#         predictions = torch.argmax(probs, dim=1).cpu().numpy()
#         trues_np = trues.flatten().cpu().numpy()

#         accuracy = cal_accuracy(predictions, trues_np)
#         f1 = f1_score(trues_np, predictions, average='weighted')
#         precision = precision_score(trues_np, predictions, average='weighted')
#         recall = recall_score(trues_np, predictions, average='weighted')

#         print('accuracy: {:.4f}'.format(adjust_float(accuracy)))
#         print('f1_score: {:.4f}'.format(adjust_float(f1)))
#         print('precision: {:.4f}'.format(adjust_float(precision)))
#         print('recall: {:.4f}'.format(adjust_float(recall)))

#         folder_path = './results/' + setting + '/'
#         os.makedirs(folder_path, exist_ok=True)
#         file_name = 'result_classification.txt'
#         with open(os.path.join(folder_path, file_name), 'a', encoding='utf-8') as f:
#             f.write(setting + "  \n")
#             f.write(f'accuracy: {adjust_float(accuracy):.4f}\n')
#             f.write(f'f1_score: {adjust_float(f1):.4f}\n')
#             f.write(f'precision: {adjust_float(precision):.4f}\n')
#             f.write(f'recall: {adjust_float(recall):.4f}\n\n')

#         if test and bool(getattr(self.args, "importance_eval", True)):
#             try:
#                 path = os.path.join("./checkpoints", setting)
#                 out_dir_imp = os.path.join(path, "importance_robustness")
#                 os.makedirs(out_dir_imp, exist_ok=True)

#                 model_eval = self.model.module if isinstance(
#                     self.model, nn.DataParallel) else self.model
#                 visualizer = ModelVisualizer(
#                     model=model_eval,
#                     device=self.device,
#                     enc_in=self.enc_in,
#                     fs=self.fs,
#                     class_names=getattr(self.args, "class_names", None),
#                 )
#                 print("[Visualization] (test-only) 运行重要性鲁棒性实验...")
#                 visualizer.run_importance_robustness_eval(
#                     data_loader=test_loader,
#                     out_dir=out_dir_imp,
#                     ratios=None,
#                     max_batches=int(
#                         getattr(self.args, "importance_eval_max_batches", -1)),
#                     use_only_correct=True,
#                     min_conf=float(
#                         getattr(self.args, "importance_eval_min_conf", 0.5)),
#                     noise_std_scale=0.5,
#                 )
#                 if hasattr(test_loader.dataset, "get_importance_gt_patch_level"):
#                     visualizer.run_importance_gt_correlation(
#                         test_loader, out_dir_imp, max_batches=10)
#             except Exception as e:
#                 print(
#                     f"[WARN] (test-only) importance_robustness_eval 失败（可忽略）: {e}")

#         return


# def adjust_float(x: float) -> float:
#     return float(np.format_float_positional(x, trim='-'))
