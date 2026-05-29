import argparse
import os
import random

import numpy as np
import torch


def str2bool(v):
    if isinstance(v, bool):
        return v
    v = str(v).lower()
    if v in ("1", "true", "t", "yes", "y"):
        return True
    if v in ("0", "false", "f", "no", "n"):
        return False
    raise argparse.ArgumentTypeError(f"Invalid bool value: {v}")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_setting(args, ii):
    return (
        f"{args.model_id}_T{args.seq_len}_D{args.d_model}_K{args.num_groups}_"
        f"Pl{args.patch_len}_S{args.stride}_{args.model}_{args.des}_{ii}"
    )


def main():
    parser = argparse.ArgumentParser(description="AnchorMoE open-source runner")
    parser.add_argument("--task_name", type=str, default="classification")
    parser.add_argument("--is_training", type=int, default=1)
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--model", type=str, default="AnchorMoE")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--des", type=str, default="main")

    parser.add_argument("--data", type=str, default="UEA")
    parser.add_argument("--root_path", type=str, required=True)
    parser.add_argument("--checkpoints", type=str, default="./checkpoints/")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--uea_norm_scope", type=str, default="train", choices=["independent", "train", "all"])
    parser.add_argument("--uea_norm_type", type=str, default="standardization")

    parser.add_argument("--seq_len", type=int, default=96)
    parser.add_argument("--enc_in", type=int, default=1)
    parser.add_argument("--num_class", type=int, default=2)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--patch_len", type=int, default=16)
    parser.add_argument("--stride", type=int, default=8)

    parser.add_argument("--use_mvre", type=str2bool, default=True)
    parser.add_argument("--mvre_num_freq_bands", type=int, default=4)
    parser.add_argument("--mvre_spectral_gate_init", type=float, default=0.0)
    parser.add_argument("--mvre_relevance_gate_init", type=float, default=-2.0)
    parser.add_argument("--num_groups", type=int, default=4)
    parser.add_argument("--router_topk", type=int, default=2)
    parser.add_argument("--router_temperature", type=float, default=1.0)
    parser.add_argument("--use_dpar", type=str2bool, default=True)
    parser.add_argument("--use_confidence", type=str2bool, default=True)
    parser.add_argument("--binding_conf_power", type=float, default=1.0)
    parser.add_argument("--composition_conf_power", type=float, default=1.0)

    parser.add_argument("--loss_type", type=str, default="ce", choices=["ce", "focal"])
    parser.add_argument("--anchor_div_lambda", type=float, default=0.05)
    parser.add_argument("--conf_lambda", type=float, default=0.01)
    parser.add_argument("--conf_warmup_epochs", type=int, default=5)
    parser.add_argument("--orth_warmup_epochs", type=int, default=5)
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--class_weight_mode", type=str, default="none", choices=["none", "balanced"])

    parser.add_argument("--itr", type=int, default=1)
    parser.add_argument("--train_epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adam", "adamw", "sgd"])
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.99)
    parser.add_argument("--adam_eps", type=float, default=1e-8)
    parser.add_argument("--sgd_momentum", type=float, default=0.9)
    parser.add_argument("--sgd_nesterov", type=str2bool, default=True)
    parser.add_argument("--scheduler", type=str, default="cawr", choices=["cawr", "cosine", "step", "none"])
    parser.add_argument("--cawr_t0", type=int, default=10)
    parser.add_argument("--cawr_tmult", type=int, default=2)
    parser.add_argument("--min_lr", type=float, default=None)
    parser.add_argument("--step_size", type=int, default=20)
    parser.add_argument("--step_gamma", type=float, default=0.5)
    parser.add_argument("--grad_clip", type=float, default=4.0)
    parser.add_argument("--aug_prob", type=float, default=0.0)
    parser.add_argument("--jitter_sigma", type=float, default=0.0)
    parser.add_argument("--scaling_sigma", type=float, default=0.0)
    parser.add_argument("--time_mask_ratio", type=float, default=0.0)
    parser.add_argument("--use_amp", action="store_true", default=False)

    parser.add_argument("--use_gpu", type=str2bool, default=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--use_multi_gpu", action="store_true", default=False)
    parser.add_argument("--devices", type=str, default="0")

    args = parser.parse_args()
    args.use_gpu = bool(torch.cuda.is_available() and args.use_gpu)
    if args.use_multi_gpu:
        args.device_ids = [int(x) for x in args.devices.replace(" ", "").split(",") if x]
        args.gpu = args.device_ids[0]
    set_seed(args.seed)

    from exp.exp_classification import ExpClassification

    for ii in range(args.itr):
        setting = build_setting(args, ii)
        exp = ExpClassification(args)
        if args.is_training:
            print(f">>>>>>> Start training: {setting}")
            exp.train(setting)
            print(f">>>>>>> Test: {setting}")
            exp.test(setting)
        else:
            exp.test(setting, test=1)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
