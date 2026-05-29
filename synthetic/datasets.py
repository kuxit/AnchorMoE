from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


SCENARIO_DISPLAY_NAMES = {
    "localized_context": "Localized-Context",
    "composition_context": "Composition-Context",
    "distractor": "Distractor",
    "multi_distractor": "Multi-Distractor",
}


CLASS_PATTERNS = {
    0: [(0, 0, 1.00), (2, 3, 0.85)],
    1: [(1, 1, 0.95), (3, 2, 0.85)],
    2: [(0, 2, -1.00), (2, 4, 0.85)],
}


COMPOSITION_PATTERNS = {
    0: [(0, 0, 0.90), (1, 1, 0.90)],
    1: [(0, 0, 0.90), (2, 2, 0.90)],
    2: [(1, 1, 0.90), (2, 2, 0.90)],
}


def motif(kind: int, length: int) -> np.ndarray:
    t = np.linspace(0, 1, length, endpoint=False)
    if kind == 0:
        return np.sin(2 * np.pi * t)
    if kind == 1:
        return np.sign(np.sin(2 * np.pi * t))
    if kind == 2:
        return 1.0 - 2.0 * np.abs(t - 0.5)
    if kind == 3:
        return np.exp(-0.5 * ((t - 0.5) / 0.16) ** 2)
    return np.sin(4 * np.pi * t)


def _valid_start(rng: np.random.RandomState, low: int, high: int, motif_len: int) -> int:
    high = max(low + 1, high - motif_len)
    return int(rng.randint(low, high))


def generate_synthetic_evidence(
    scenario: str,
    n_samples: int = 1000,
    seq_len: int = 128,
    channels: int = 4,
    num_classes: int = 3,
    motif_len: int = 16,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic evidence-localization data.

    Returns:
        x: float32 array of shape (N, T, C)
        y: int64 class labels of shape (N,)
        gt_mask: binary temporal evidence mask of shape (N, T)
    """
    if scenario not in SCENARIO_DISPLAY_NAMES:
        raise ValueError(f"Unknown scenario {scenario!r}. Available: {list(SCENARIO_DISPLAY_NAMES)}")

    rng = np.random.RandomState(seed)
    x = rng.normal(0.0, 0.18, size=(n_samples, seq_len, channels)).astype(np.float32)
    y = rng.randint(0, num_classes, size=(n_samples,), dtype=np.int64)
    gt = np.zeros((n_samples, seq_len), dtype=np.float32)

    for i in range(n_samples):
        cls = int(y[i])
        patterns = COMPOSITION_PATTERNS[cls] if scenario == "composition_context" else CLASS_PATTERNS[cls]

        if scenario == "localized_context":
            spans = [(_valid_start(rng, 8, seq_len - 8, motif_len), patterns[0])]
        else:
            spans = [
                (_valid_start(rng, 8, seq_len // 2 - 4, motif_len), patterns[0]),
                (_valid_start(rng, seq_len // 2 + 4, seq_len - 8, motif_len), patterns[1]),
            ]

        for start, (channel, kind, amp) in spans:
            end = min(seq_len, start + motif_len)
            x[i, start:end, channel] += float(amp) * motif(kind, end - start).astype(np.float32)
            gt[i, start:end] = 1.0

        if scenario in {"localized_context", "composition_context", "distractor", "multi_distractor"}:
            n_distractors = {
                "localized_context": 1,
                "composition_context": 1,
                "distractor": 1,
                "multi_distractor": 3,
            }[scenario]
            amp = {
                "localized_context": 1.2,
                "composition_context": 1.2,
                "distractor": 1.8,
                "multi_distractor": 2.2,
            }[scenario]
            for _ in range(n_distractors):
                d_start = _valid_start(rng, 8, seq_len - 8, motif_len)
                d_channel = int(rng.randint(0, channels))
                d_kind = int(rng.randint(0, 5))
                x[i, d_start : d_start + motif_len, d_channel] += (
                    amp * motif(d_kind, motif_len).astype(np.float32)
                )

        if scenario in {"localized_context", "composition_context"}:
            context = 0.08 if y[i] == 0 else (-0.08 if y[i] == 1 else 0.04)
            x[i, :, -1] += context

    return x, y, gt


def save_synthetic_npz(
    out_dir: str | Path,
    scenario: str,
    train_size: int = 900,
    test_size: int = 300,
    seq_len: int = 128,
    channels: int = 4,
    num_classes: int = 3,
    motif_len: int = 16,
    seed: int = 42,
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    x_train, y_train, gt_train = generate_synthetic_evidence(
        scenario, train_size, seq_len, channels, num_classes, motif_len, seed + 1
    )
    x_test, y_test, gt_test = generate_synthetic_evidence(
        scenario, test_size, seq_len, channels, num_classes, motif_len, seed + 2
    )
    np.savez_compressed(out_dir / f"{scenario}_train.npz", x=x_train, y=y_train, gt_mask=gt_train)
    np.savez_compressed(out_dir / f"{scenario}_test.npz", x=x_test, y=y_test, gt_mask=gt_test)


class SyntheticEvidenceDataset(Dataset):
    """Torch dataset wrapper for the four synthetic evidence scenarios."""

    def __init__(
        self,
        scenario: str,
        split: str = "train",
        train_size: int = 900,
        test_size: int = 300,
        seq_len: int = 128,
        channels: int = 4,
        num_classes: int = 3,
        motif_len: int = 16,
        seed: int = 42,
    ):
        n = train_size if split.lower() == "train" else test_size
        split_seed = seed + (1 if split.lower() == "train" else 2)
        x, y, gt = generate_synthetic_evidence(
            scenario, n, seq_len, channels, num_classes, motif_len, split_seed
        )
        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y)
        self.gt_mask = torch.from_numpy(gt)

    def __len__(self) -> int:
        return int(self.y.numel())

    def __getitem__(self, index: int):
        return self.x[index], self.y[index], self.gt_mask[index]
