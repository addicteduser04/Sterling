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
  --output-dir outputs_corrected
```

The checked-in legacy `data/masi/bam_ecb_2004.csv` currently contains BAM in
percentage points too. Until that file is regenerated from decimal BAM input,
use `--bam-unit percent` for that particular file. The standalone BAM path in
the example is not included in every checkout; omit `--bam-data` only if using
the aligned BAM calendar is intentional.
