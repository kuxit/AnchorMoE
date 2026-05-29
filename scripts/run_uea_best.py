from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


PASS_KEYS = [
    "seq_len",
    "d_model",
    "num_groups",
    "patch_len",
    "stride",
    "dropout",
    "batch_size",
    "learning_rate",
    "train_epochs",
    "patience",
    "optimizer",
    "scheduler",
    "weight_decay",
    "label_smoothing",
    "loss_type",
    "class_weight_mode",
    "focal_gamma",
    "anchor_div_lambda",
    "conf_lambda",
    "router_temperature",
    "router_topk",
    "binding_conf_power",
    "composition_conf_power",
    "aug_prob",
    "jitter_sigma",
    "scaling_sigma",
    "time_mask_ratio",
    "seed",
    "uea_norm_scope",
]


def load_configs(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload["datasets"]


def build_command(dataset: str, cfg: dict, args: argparse.Namespace) -> list[str]:
    data_root = Path(args.data_root)
    root_path = data_root / dataset
    cmd = [
        sys.executable,
        str(ROOT / "run.py"),
        "--task_name",
        "classification",
        "--is_training",
        "1",
        "--model",
        "AnchorMoE",
        "--model_id",
        dataset,
        "--data",
        "UEA",
        "--root_path",
        str(root_path),
        "--des",
        args.des,
        "--checkpoints",
        args.checkpoints,
        "--num_workers",
        str(args.num_workers),
        "--use_gpu",
        str(args.use_gpu).lower(),
        "--gpu",
        str(args.gpu),
    ]
    for key in PASS_KEYS:
        if key in cfg and cfg[key] is not None:
            cmd.extend([f"--{key}", str(cfg[key])])
    if args.extra:
        cmd.extend(args.extra)
    return cmd


def main():
    parser = argparse.ArgumentParser(description="Run AnchorMoE with stored best UEA configs.")
    parser.add_argument("--config", type=str, default=str(ROOT / "configs" / "uea_best_configs.json"))
    parser.add_argument("--datasets", type=str, default="all", help="Comma-separated dataset names or 'all'.")
    parser.add_argument("--data_root", type=str, default=str(ROOT / "data" / "UEA" / "Multivariate_ts"))
    parser.add_argument("--checkpoints", type=str, default="./checkpoints_best/")
    parser.add_argument("--des", type=str, default="best_config")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--use_gpu", type=str, default="true")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("extra", nargs=argparse.REMAINDER, help="Extra arguments passed to run.py after '--'.")
    args = parser.parse_args()

    if args.extra and args.extra[0] == "--":
        args.extra = args.extra[1:]
    args.use_gpu = str(args.use_gpu).lower() in {"1", "true", "yes", "y"}

    configs = load_configs(Path(args.config))
    datasets = list(configs) if args.datasets.lower() == "all" else [
        x.strip() for x in args.datasets.split(",") if x.strip()
    ]

    for dataset in datasets:
        if dataset not in configs:
            raise KeyError(f"Dataset {dataset!r} not found in {args.config}")
        cfg = configs[dataset]
        print(
            f"[{dataset}] reported_acc={cfg.get('reported_accuracy')} "
            f"reported_f1={cfg.get('reported_f1')} trial={cfg.get('trial_name')}",
            flush=True,
        )
        cmd = build_command(dataset, cfg, args)
        print(" ".join(f'"{x}"' if " " in x else x for x in cmd), flush=True)
        if not args.dry_run:
            subprocess.run(cmd, cwd=str(ROOT), check=True)


if __name__ == "__main__":
    main()
