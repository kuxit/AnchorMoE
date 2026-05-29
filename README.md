# AnchorMoE

This repository contains the open-source implementation of AnchorMoE for UEA multivariate time-series classification.

## Main Components

- **MVRE**: Multi-View Representation Embedding for local evidence units.
- **DPAR**: Diversified Posterior-Anchor Routing for evidence type decomposition.
- **RGC**: Reliability-Gated Composition for additive prediction and intrinsic explanation.

The main implementation is in `models/AnchorMoE.py`, and the training objective is in `loss.py`.

## Directory

```text
open_source_anchor_moe/
  data/UEA/Multivariate_ts/     # place downloaded UEA datasets here
  data_provider/                # UEA loader
  exp/                          # classification training loop
  models/AnchorMoE.py           # AnchorMoE model
  synthetic/                    # synthetic evidence dataset generators
  utils/                        # minimal utilities
  loss.py                       # classification + orthogonality + reliability losses
  run.py                        # entry point
  scripts/run_example.ps1       # example command
  scripts/generate_synthetic_data.py
  configs/uea_best_configs.json # stored per-dataset best known configs
```

## Installation

```bash
pip install -r requirements.txt
```

The code was tested with Python 3.10+ and PyTorch.

## Example

```bash
python run.py \
  --task_name classification \
  --is_training 1 \
  --model AnchorMoE \
  --model_id BasicMotions \
  --data UEA \
  --root_path ./data/UEA/Multivariate_ts/BasicMotions \
  --seq_len 100 \
  --d_model 128 \
  --num_groups 4 \
  --patch_len 8 \
  --stride 4 \
  --batch_size 16 \
  --train_epochs 50
```

The official UEA TRAIN split is used for training. The official TEST split is used for validation and final testing, matching the experimental protocol used in this project.

UEA data files are not included in this repository. Download the official UEA multivariate archive and place each dataset under:

```text
data/UEA/Multivariate_ts/<DatasetName>/<DatasetName>_TRAIN.ts
data/UEA/Multivariate_ts/<DatasetName>/<DatasetName>_TEST.ts
```

## Reproduce Best-Known UEA Configurations

Per-dataset configurations are stored in `configs/uea_best_configs.json`.
To run one dataset:

```bash
python scripts/run_uea_best.py --datasets BasicMotions
```

To run all 29 UEA datasets:

```bash
python scripts/run_uea_best.py --datasets all
```

The config file also records the locally reported Accuracy/F1. Re-training can show small variation across seeds, hardware, and library versions.

## Output

Training writes checkpoints to `checkpoints/` and final metrics to `results/`.

The model can return intrinsic explanation quantities by calling:

```python
logits, aux = model(x, x_mark_enc=padding_mask, return_aux=True)
```

Important entries include:

- `routing_probs`: DPAR patch-to-expert routing weights.
- `posterior_anchors`: routed posterior anchors.
- `patch_confidence`: RGC patch-level reliability score.
- `patch_importance`: normalized patch composition weight.
- `signed_patch_contribution`: signed contribution for the predicted class.
- `patch_contribution`: additive patch-level class contribution.

## Synthetic Evidence Datasets

The release also includes generators for the four synthetic evidence-localization datasets used in the paper:

- Localized-Context
- Composition-Context
- Distractor
- Multi-Distractor

See `SYNTHETIC_DATASETS.md` for details. To generate reproducible `.npz` files:

```bash
python scripts/generate_synthetic_data.py --out_dir ./data/synthetic
```
