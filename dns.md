# BAM yield-curve modelling

The active research pipeline is split into three modules:

- `src/modelling/dns.py`: persistence and Dynamic Nelson–Siegel/Kalman/OU
  controls, including the leakage-safe persistence–DNS blend.
- `src/modelling/phase1.py`: direct NS/PCA factor-change models and the
  regularized PCA VAR.
- `src/modelling/phase2.py`: direct maturity models, XGBoost factor models,
  overlap-derived DM inference, and weekly robustness.

All rates are decimals internally (`0.035 = 3.5%`). Input units are explicit;
plots alone convert to percentages or basis points.

## Leakage rules

At each forecast origin, model fitting, scaling, Nelson–Siegel calibration,
PCA, features, and hyperparameter selection use only historically available
observations. A direct horizon-h target is admitted only once its target date
has occurred. Expanding validation folds are purged for target overlap.

DM/Newey–West bandwidths are derived from actual origin-to-target forecast
windows. For the monthly experiment they are 0 at J+5/J+10 and 1 at J+22.

## Commands

```bash
python src/modelling/dns.py \
  --combined-data data/masi/bam_ecb_2004.csv \
  --bam-unit percent --ecb-unit percent \
  --output-dir outputs_dns

python -m src.modelling.phase1 \
  --bam-data data/masi/bam_ecb_2004.csv \
  --bam-unit percent --output-dir outputs_phase1

python -m src.modelling.phase2 \
  --bam-data data/masi/bam_ecb_2004.csv \
  --bam-unit percent --phase1-dir outputs_phase1 \
  --output-dir outputs_phase2
```

The legacy `data/masi/bam_ecb_2004.csv` stores both BAM and ECB inputs in
percentage points, hence the explicit `percent` arguments above.

## Validated results

- `outputs_weighted/`: retained DNS and persistence–DNS controls.
- `outputs_phase1/`: regularized NS/PCA experiments.
- `outputs_phase2/`: direct maturity, XGBoost, corrected DM, stability, and
  weekly robustness results.

The current evidence does not establish a statistically robust aggregate
improvement over persistence. See `outputs_phase2/phase2_report.md` for the
latest scientific conclusion.
