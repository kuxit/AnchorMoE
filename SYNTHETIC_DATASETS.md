# Synthetic Evidence Datasets

The open-source release includes generators for four synthetic datasets used for evidence-localization evaluation. Each sample is a multivariate time series with a binary ground-truth temporal evidence mask.

Default setting:

- sequence length: 128
- channels: 4
- classes: 3
- motif length: 16
- training samples: 900
- test samples: 300
- base noise: Gaussian noise with standard deviation 0.18

## Datasets

| Dataset | Purpose | Ground-truth evidence | Distractors |
|---|---|---|---|
| Localized-Context | Tests whether an explanation can recover one local class-discriminative motif under weak sample-level context. | One class-specific local motif. | One label-independent motif plus weak global context. |
| Composition-Context | Tests whether an explanation can recover multiple local evidence regions whose combination defines the class. | Two local motifs; individual motifs are intentionally ambiguous across classes. | One label-independent motif plus weak global context. |
| Distractor | Tests robustness to one strong but label-irrelevant local pattern. | Two class-specific local motifs. | One high-amplitude label-independent motif. |
| Multi-Distractor | Tests robustness to multiple strong spurious local patterns. | Two class-specific local motifs. | Three high-amplitude label-independent motifs. |

Distractor regions are visually salient but independent of the class label, so they are not marked as evidence in the ground-truth mask.

## Generate NPZ Files

```bash
python scripts/generate_synthetic_data.py --out_dir ./data/synthetic
```

This creates:

```text
localized_context_train.npz
localized_context_test.npz
composition_context_train.npz
composition_context_test.npz
distractor_train.npz
distractor_test.npz
multi_distractor_train.npz
multi_distractor_test.npz
```

Each `.npz` contains:

- `x`: time series, shape `(N, T, C)`
- `y`: class labels, shape `(N,)`
- `gt_mask`: binary evidence mask, shape `(N, T)`

