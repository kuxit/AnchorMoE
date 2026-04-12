import argparse
import os
import random
import numpy as np
import torch


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    v = str(v).lower()
    if v in ("1", "true", "t", "yes", "y"):
        return True
    if v in ("0", "false", "f", "no", "n"):
        return False
    raise argparse.ArgumentTypeError(f"Invalid bool value: {v}")


def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def configure_device(args):
    args.use_gpu = bool(torch.cuda.is_available() and args.use_gpu)
    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(" ", "")
        device_ids = args.devices.split(",")
        args.device_ids = [int(i) for i in device_ids if i != ""]
        if len(args.device_ids) == 0:
            raise ValueError("use_multi_gpu=True but --devices is empty/invalid")
        args.gpu = args.device_ids[0]
    return args


def get_exp_class(task_name: str):
    if task_name == "classification":
        from exp.exp_classification import Exp_Classification
        return Exp_Classification
    if task_name == "long_term_forecast":
        from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
        return Exp_Long_Term_Forecast
    if task_name == "anomaly_detection":
        from exp.exp_anomaly_detection import Exp_Anomaly_Detection
        return Exp_Anomaly_Detection
    raise ValueError(f"Unknown task_name: {task_name}")


def build_setting(args, ii: int) -> str:
    if args.model in ("newmodel", "AnchorMoE"):
        return (
            f"{args.task_name}_{args.model_id}_"
            f"T{args.seq_len}_D{args.d_model}_Groups{args.num_groups}_"
            f"Pl{args.patch_len}_S{args.stride}_"
            f"{args.model}_{args.des}_{ii}"
        )

    if args.model == "newmodel_ablation":
        ablation_type = getattr(args, "ablation_type", "full")
        return (
            f"{args.task_name}_{args.model_id}_"
            f"T{args.seq_len}_D{args.d_model}_Groups{args.num_groups}_"
            f"Pl{args.patch_len}_S{args.stride}_"
            f"ablation_{ablation_type}_{args.des}_{ii}"
        )

    return (
        f"{args.task_name}_{args.model_id}_{args.comment if hasattr(args, 'comment') else ''}_{args.model}_{args.data}_"
        f"T{args.seq_len}_C{args.enc_in}_{args.des}_{ii}"
    )


def main():
    parser = argparse.ArgumentParser(description="NewModel Time-Series Runner")

    parser.add_argument("--num_groups", type=int, default=4, help="Number of expert groups")
    parser.add_argument("--anchor_div_lambda", type=float, default=0.05,
                        help="Anchor diversity regularization weight")
    parser.add_argument("--conf_lambda", type=float, default=0.0,
                        help="UADB confidence supervision weight")
    parser.add_argument("--conf_alpha", type=float, default=3.0,
                        help="UADB confidence target sharpness in exp(-alpha * s_l)")
    parser.add_argument("--conf_warmup_epochs", type=int, default=0,
                        help="Linear warmup epochs for confidence supervision weight")
    parser.add_argument("--orth_warmup_epochs", type=int, default=0,
                        help="Linear warmup epochs for orthogonality regularization weight")
    parser.add_argument("--expert_div_lambda", type=float, default=0.05,
                        help="Specialist diversity regularization weight")
    parser.add_argument("--expert_div_warmup_epochs", type=int, default=0,
                        help="Linear warmup epochs for specialist diversity regularization weight")
    parser.add_argument("--log_conf_diagnostics", type=str2bool, default=False,
                        help="Compute and log confidence diagnostics even when conf_lambda=0")
    parser.add_argument("--spectral_weight", type=float, default=0.3,
                        help="Spectral-view weight inside the dual-view router")
    parser.add_argument("--use_relevance_query", type=str2bool, default=True,
                        help="Whether to include the sample-level relevance cue in the routing query")
    parser.add_argument("--use_confidence", type=str2bool, default=True,
                        help="Whether to enable expert confidence weighting")
    parser.add_argument("--enable_viz", type=str2bool, default=True,
                        help="Whether to run visualization after training")
    parser.add_argument("--importance_eval", type=str2bool, default=True,
                        help="Whether to run importance robustness evaluation")
    parser.add_argument("--importance_eval_max_batches", type=int, default=-1,
                        help="Maximum number of batches used by importance evaluation")
    parser.add_argument("--importance_eval_min_conf", type=float, default=0.5,
                        help="Minimum p(true_class) for importance evaluation samples")
    parser.add_argument("--posthoc_compare_eval", type=str2bool, default=False,
                        help="Whether to benchmark intrinsic explanations against post-hoc methods")
    parser.add_argument("--posthoc_compare_max_samples", type=int, default=64,
                        help="Maximum number of samples used by intrinsic-vs-posthoc benchmarking")
    parser.add_argument("--posthoc_compare_methods", type=str, default="intrinsic,grad,input_x_grad",
                        help="Comma-separated explanation methods to benchmark")
    parser.add_argument("--aopc_eval", type=str2bool, default=False,
                        help="Whether to run intrinsic AOPC evaluation on the test split")
    parser.add_argument("--aopc_max_samples", type=int, default=-1,
                        help="Maximum number of samples used by AOPC; -1 means full eligible test set")
    parser.add_argument("--aopc_perturb_mode", type=str, default="zero",
                        choices=["zero", "mean"],
                        help="Perturbation used by AOPC")
    parser.add_argument("--iou_eval", type=str2bool, default=False,
                        help="Whether to run top-k patch IoU evaluation on the test split")
    parser.add_argument("--iou_max_samples", type=int, default=128,
                        help="Maximum number of samples used by IoU evaluation")
    parser.add_argument("--stage1_internal_eval", type=str2bool, default=False,
                        help="Whether to log internal-mechanism and efficiency statistics")
    parser.add_argument("--stage1_log_dir", type=str, default="stage1_internal_mechanism_eval",
                        help="Subfolder name used to store stage1 internal statistics")
    parser.add_argument("--stage1_active_threshold", type=float, default=0.01,
                        help="Expert is active when its token-mass share in a batch exceeds this threshold")
    parser.add_argument("--stage1_kappa_low", type=float, default=0.1,
                        help="Threshold used for low-kappa ratio statistics")
    parser.add_argument("--stage1_kappa_high", type=float, default=0.85,
                        help="Threshold used for high-kappa ratio statistics")
    parser.add_argument("--resume", type=str2bool, default=True,
                        help="Resume unfinished runs from the latest resume_state.pth if available")
    parser.add_argument("--skip_completed", type=str2bool, default=False,
                        help="Skip a run when final test metrics already exist")

    parser.add_argument("--task_name", type=str, required=True, default="classification")
    parser.add_argument("--is_training", type=int, required=True, default=1)
    parser.add_argument("--model_id", type=str, required=True, default="test")
    parser.add_argument("--model", type=str, required=True, default="newmodel",
                        help="Model name, e.g. newmodel or AnchorMoE")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--des", type=str, default="test")

    parser.add_argument("--data", type=str, required=True, default="UEA")
    parser.add_argument("--root_path", type=str, default="./dataset/UEA/")
    parser.add_argument("--checkpoints", type=str, default="./checkpoints/")
    parser.add_argument("--num_workers", type=int, default=10)

    parser.add_argument("--patch_len", type=int, default=16, help="Patch length")
    parser.add_argument("--stride", type=int, default=8, help="Patch stride")

    parser.add_argument("--seq_len", type=int, default=96)
    parser.add_argument("--enc_in", type=int, default=7)
    parser.add_argument("--num_class", type=int, default=10)

    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--itr", type=int, default=1)
    parser.add_argument("--train_epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--use_amp", action="store_true", default=False)

    parser.add_argument("--use_gpu", type=bool, default=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--use_multi_gpu", action="store_true", default=False)
    parser.add_argument("--devices", type=str, default="0")
    parser.add_argument("--gpu_type", type=str, default="cuda", help="gpu type: cuda or mps")
    args = parser.parse_args()
    args.amp = args.use_amp

    args = configure_device(args)
    set_all_seeds(int(args.seed))

    Exp = get_exp_class(args.task_name)

    if args.is_training:
        for ii in range(args.itr):
            setting = build_setting(args, ii)
            result_file = f"./results/{setting}/result_classification.txt"
            if bool(getattr(args, "skip_completed", False)) and os.path.exists(result_file):
                print(f">>>>>>> Skipping completed run : {setting} >>>>>>>>>>>>>>>>>>>>>>>>>>")
                continue
            exp = Exp(args)
            print(f">>>>>>> Start Training : {setting} >>>>>>>>>>>>>>>>>>>>>>>>>>")
            exp.train(setting)
            print(f">>>>>>> Testing : {setting} <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
            exp.test(setting)
            torch.cuda.empty_cache()
    else:
        ii = 0
        setting = build_setting(args, ii)
        exp = Exp(args)
        print(f">>>>>>> Testing : {setting} <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        exp.test(setting, test=1)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
