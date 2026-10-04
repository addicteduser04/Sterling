# Modélisation de Nelson–Siegel

A quantitative finance research that aims to predict the 3 parameters of the Nelsen-Siegel beta0, beta1, and beta2 using Time Series models, Machine Learning and Deep Learning.

## DNS–Kalman rate units

`src/modelling/dns.py` uses one internal convention: every BAM and ECB yield is
a decimal rate (`0.035` means 3.5%). Unit conversion is explicit and occurs
once, immediately after CSV loading and before interpolation, merging, factor
extraction, calibration, forecasting, or evaluation:

- `--bam-unit decimal` is the default and leaves BAM unchanged;
- `--ecb-unit percent` is the default and divides ECB values by 100;
- plots multiply rates, rate factors, and rate errors by 100 for presentation;
- saved factor, fitted-yield, forecast, and backtest CSV files remain decimal
  and include a `rate_unit` column; p-values and dimensionless diagnostics are
  never rescaled.

The loader never guesses units from a rolling window. It rejects nonnumeric or
infinite selected rates and implausible post-normalisation magnitudes while
allowing supported missing observations and legitimate negative ECB rates.

Run the pipeline with:

```bash
python src/modelling/dns.py \
  --combined-data data/masi/bam_ecb_2004.csv \
  --bam-data data/processed/bam_observed_and_interpolated.csv \
  --bam-unit decimal \
  --ecb-unit percent \
  --output-dir outputs_dns
```

The checked-in legacy `data/masi/bam_ecb_2004.csv` currently contains BAM in
percentage points too. Until that file is regenerated from decimal BAM input,
use `--bam-unit percent` for that particular file. The standalone BAM path in
the example is not included in every checkout; omit `--bam-data` only if using
the aligned BAM calendar is intentional.

## Phase 1: direct change predictability

The leakage-safe Phase 1 experiment asks whether direct J+5, J+10 and J+22
changes can beat a zero-change persistence forecast at the nine observed BAM
maturities. It includes `NS_RIDGE`, `NS_ELASTICNET`, origin-local PCA,
`PCA_RIDGE`, and a parsimonious `PCA_RIDGE_VAR`. PCA, feature scaling,
hyperparameter tuning and direct targets are all rebuilt at each expanding
forecast origin. Inner validation is purged so training targets are already
known at the first validation origin.

```bash
python -m src.modelling.phase1 \
  --bam-data data/masi/bam_ecb_2004.csv \
  --bam-unit percent \
  --output-dir outputs_phase1
```

By default, this attaches the existing leakage-safe DNS and DNS-blend forecasts
from `outputs_weighted/backtest_blended_bam.csv` only on exact common rows. Use
`--rerun-dns` to recalibrate that benchmark. Outputs include long-form curve and
factor forecasts, sample-size and PCA audits, metrics, robust DM comparisons,
figures, a master table, and `phase1_report.md`.

## Phase 2: maturity-specific and nonlinear dynamics

Phase 2 audits Phase 1 DM bandwidths from actual forecast-window overlap, adds
separate `DIRECT_RIDGE` and `DIRECT_AR` maturity models, and evaluates
`NS_XGBOOST` and `PCA_XGBOOST` with three deliberately conservative tree
configurations. It also runs a separate weekly-origin robustness experiment
from 2024 onward. `DIRECT_ARIMA` is explicitly rejected because ordinary ARIMA
assumes equally spaced observations while BAM publication dates are irregular.

```bash
python -m src.modelling.phase2 \
  --bam-data data/masi/bam_ecb_2004.csv \
  --bam-unit percent \
  --phase1-dir outputs_phase1 \
  --output-dir outputs_phase2
```

The full command is computationally intensive because every PCA, NS and model
fit remains origin-local. Use `--skip-weekly` only for development runs. The
scientific report is written to `outputs_phase2/phase2_report.md`.

## Phase 3A: real-time macro data audit

Run `python -m src.modelling.phase3_data --project-dir . --output-dir outputs_phase3a`
before modelling. It generates the catalogue tables, accepted policy-rate
observations, checksums, and sample accounting. See
`docs/PHASE3A_MACRO_DATA_AUDIT.md`; monthly series are joined by their actual
`available_from` date, never their reference month.
