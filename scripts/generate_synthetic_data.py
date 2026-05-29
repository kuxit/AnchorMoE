import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from synthetic.datasets import SCENARIO_DISPLAY_NAMES, save_synthetic_npz


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic evidence-localization datasets.")
    parser.add_argument("--out_dir", type=str, default="./data/synthetic")
    parser.add_argument("--train_size", type=int, default=900)
    parser.add_argument("--test_size", type=int, default=300)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--channels", type=int, default=4)
    parser.add_argument("--num_classes", type=int, default=3)
    parser.add_argument("--motif_len", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    for scenario in SCENARIO_DISPLAY_NAMES:
        save_synthetic_npz(
            out_dir,
            scenario,
            train_size=args.train_size,
            test_size=args.test_size,
            seq_len=args.seq_len,
            channels=args.channels,
            num_classes=args.num_classes,
            motif_len=args.motif_len,
            seed=args.seed,
        )
        print(f"saved {scenario} -> {out_dir}")


if __name__ == "__main__":
    main()
