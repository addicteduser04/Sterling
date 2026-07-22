# Dynamic Nelson-Siegel + Kalman Filter

This note explains the implementation in [src/modelling/DNS.py](src/modelling/DNS.py) step by step.

## 1. Goal

The script fits a Dynamic Nelson-Siegel (DNS) model to the BAM yield curve data, using the maturity columns ending in `_x` from [data/masi/bam_ecb_2004.csv](data/masi/bam_ecb_2004.csv).

The goal is to estimate three latent factors:
- `beta0`: level factor
- `beta1`: slope factor
- `beta2`: curvature factor

These factors are then used to reconstruct the observed yield curve.

## 2. Data loading

The script starts by reading the CSV file and selecting only the BAM yield columns:
- columns ending in `_x`

It converts the index to a datetime index and sorts the data chronologically.

## 3. Maturity parsing

Each column name is converted into a maturity in years:
- `3M_x` becomes `0.25`
- `6M_x` becomes `0.5`
- `1Y_x` becomes `1.0`
- `10Y_x` becomes `10.0`

This is done by the `parse_maturity` helper.

## 4. Nelson-Siegel loadings

The Nelson-Siegel model expresses yields as:

$$y(\tau) = \beta_0 + \beta_1 \cdot \frac{1 - e^{-\lambda \tau}}{\lambda \tau} + \beta_2 \left( \frac{1 - e^{-\lambda \tau}}{\lambda \tau} - e^{-\lambda \tau} \right)$$

The function `nelson_siegel_loadings` builds the loading matrix for each maturity `tau`.

This matrix is later used to connect latent factors to observed yields.

## 5. Kalman filter setup

The model assumes that the latent factors follow a simple random-walk process:

- state at time $t$ depends on the previous state
- small process noise is added
- observations are noisy measurements of the yield curve

The script initializes:
- a zero mean for the state vector
- an identity covariance matrix
- a small process noise term
- a small observation noise term

## 6. Filtering step

For each date in the yield curve data, the script:
1. reads the observed yields for that date
2. selects the valid observations
3. predicts the next state using the previous estimate
4. compares the prediction with the observed yields
5. updates the factor estimates using the Kalman gain

This produces a filtered estimate of `beta0`, `beta1`, and `beta2` for each time step.

## 7. Reconstructing fitted yields

Once the factors are estimated, the script reconstructs the fitted yield curves using the Nelson-Siegel loadings.

This gives a model-based version of the observed yields, which can be compared to the real data.

## 8. Forecasting

If a forecast horizon is requested, the script can project the factors forward for a few periods.

These forecasted factors are then transformed into forecasted yields using the same Nelson-Siegel structure.

## 9. Outputs

The script writes the following files to [data/analysis_results](data/analysis_results):
- `dns_kalman_betas.csv`: estimated factor values
- `dns_kalman_fitted_yields.csv`: fitted yields from the filtered factors
- `dns_kalman_forecast_betas.csv`: forecasted factors
- `dns_kalman_forecast_yields.csv`: forecasted yields

## 10. How to run it

From the project root, run:

```bash
python src/modelling/DNS.py
```

This will generate the output CSV files automatically.
