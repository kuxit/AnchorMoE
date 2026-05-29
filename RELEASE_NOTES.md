# AnchorMoE v1.0.0

Initial open-source release of AnchorMoE.

## Included

- AnchorMoE model implementation with MVRE, DPAR, and RGC.
- Training and evaluation runner for UEA multivariate time-series classification.
- Per-dataset best-known UEA configuration file.
- Synthetic evidence dataset generators for interpretability experiments.
- Minimal UEA data loader and experiment loop.

## Not Included

- UEA data files are not included. Please download the official UEA multivariate archive and place datasets under `data/UEA/Multivariate_ts/`.
- Checkpoints and experiment logs are not included.

## Reproduction

Run a stored best-known configuration:

```bash
python scripts/run_uea_best.py --datasets BasicMotions
```

Generate synthetic evidence datasets:

```bash
python scripts/generate_synthetic_data.py --out_dir ./data/synthetic
```
