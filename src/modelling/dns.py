"""Pipeline DNS--Kalman BAM/BCE chronologique et reproductible.

Cette version remplace le prototype historique. Principes essentiels :

* ``decay_time`` est la constante de temps de ``exp(-tau / decay_time)`` ;
* tous les paramètres d'une origine de backtest sont estimés sur son passé ;
* les dynamiques BAM et BCE sont estimées séparément ;
* les écarts irréguliers entre dates modifient la transition d'état ;
* l'influence BCE est centrée, régularisée et strictement retardée ;
* la première prévision BAM à T+1 utilise BCE(T), connue à T ;
* l'évaluation principale porte sur les taux observés et la persistance ;
* les facteurs OLS ne sont conservés que comme diagnostics secondaires.
* l'unité interne unique est le taux décimal (0.035 = 3.5 %) ;
* les données BAM sont décimales et les données BCE en points de pourcentage
  sont divisées par 100 une seule fois, dès leur chargement ;
* la multiplication par 100 est réservée aux graphiques.

Exemple
-------
python src/modelling/dns.py \
  --combined-data data/masi/bam_ecb_2004.csv \
  --bam-data data/processed/bam_observed_and_interpolated.csv \
  --bam-unit decimal --ecb-unit percent \
  --output-dir outputs_corrected

Si ``--bam-data`` est omis, le modèle BAM seul utilise le calendrier du panel
aligné BAM--BCE. Le programme l'indique explicitement dans son rapport.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve
from scipy.stats import t as student_t


FACTOR_NAMES = ("beta0", "beta1", "beta2")
FACTOR_LABELS = ("β₀ (niveau)", "β₁ (court − long)", "β₂ (courbure)")
FACTOR_COLORS = ("#1f77b4", "#2ca02c", "#d62728")
OBSERVED_BAM_MATURITIES = np.array([0.25, 0.5, 1, 2, 5, 10, 15, 20, 30])
EPS = 1e-10
PLOT_PERCENT = 100.0
VALID_RATE_UNITS = ("decimal", "percent")


@dataclass(frozen=True)
class OUParameters:
    """Paramètres d'une dynamique de retour à la moyenne en temps continu."""

    long_run_mean: np.ndarray
    kappa: np.ndarray
    diffusion_cov: np.ndarray


@dataclass(frozen=True)
class DNSModel:
    decay_time: float
    loadings: np.ndarray
    dynamics: OUParameters
    measurement_cov: np.ndarray
    init_mean: np.ndarray
    init_cov: np.ndarray
    condition_number: float


@dataclass
class FilterResult:
    filtered: np.ndarray
    predicted_next: np.ndarray
    covariances: np.ndarray
    log_likelihood: float


@dataclass
class BacktestOrigin:
    month: pd.Timestamp
    cutoff_date: pd.Timestamp
    future_dates: pd.DatetimeIndex
    decay_time_bam: float
    decay_time_ecb: float | None
    predicted_betas: np.ndarray
    true_betas_ols: np.ndarray
    predicted_yields: np.ndarray
    true_yields: np.ndarray
    persistence_yields: np.ndarray
    rmse_all: float
    mae_all: float
    rmse_observed: float
    mae_observed: float
    rmse_persistence_observed: float
    mse_observed: float
    mse_persistence_observed: float
    rmse_by_maturity: np.ndarray
    rmse_beta: np.ndarray


@dataclass(frozen=True)
class PipelineConfig:
    """Configuration importable; all rates are decimal after ingestion."""

    combined_data: str | Path
    output_dir: str | Path = "outputs_corrected"
    bam_data: str | Path | None = None
    bam_unit: str = "decimal"
    ecb_unit: str = "percent"
    start_date: str = "2022-01-01"
    horizon: int = 22
    interpolated_weight: float = 0.35
    influence_window: int = 500
    influence_ridge: float = 10.0
    skip_backtest: bool = False


# ---------------------------------------------------------------------------
# Données et maturités
# ---------------------------------------------------------------------------


def _read_numeric_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {path}")
    df = pd.read_csv(path, index_col=0)
    df.index = pd.to_datetime(df.index, errors="coerce")
    if df.index.isna().any():
        raise ValueError(f"Dates non reconnues dans {path}")
    df = df.sort_index()
    if df.index.has_duplicates:
        warnings.warn(
            f"{df.index.duplicated(keep=False).sum()} lignes portent une date "
            "dupliquée ; la dernière occurrence est conservée.",
            stacklevel=2,
        )
        df = df[~df.index.duplicated(keep="last")]
    return df


def normalize_rate_units(
    df: pd.DataFrame,
    input_unit: str,
    source_name: str,
) -> pd.DataFrame:
    """Convertit explicitement les taux vers l'unité interne décimale.

    Aucune détection heuristique n'est utilisée : l'unité du fichier est un
    paramètre déclaré. ``decimal`` signifie que 3,5 % est stocké sous 0.035 ;
    ``percent`` signifie qu'il est stocké sous 3.5.
    """

    if input_unit not in VALID_RATE_UNITS:
        raise ValueError(
            f"Unité inconnue pour {source_name}: {input_unit!r}. "
            f"Valeurs admises: {VALID_RATE_UNITS}."
        )
    numeric = df.apply(pd.to_numeric, errors="coerce")
    invalid = df.notna() & numeric.isna()
    if invalid.any().any():
        locations = [
            f"{column}@{index}"
            for index, column in invalid.stack()[lambda values: values].index[:5]
        ]
        raise ValueError(
            f"{source_name}: taux non numériques dans " + ", ".join(locations) + "."
        )
    normalized = numeric.astype(float).copy()
    if input_unit == "percent":
        normalized /= 100.0

    values = normalized.to_numpy(float)
    if np.isinf(values).any():
        raise ValueError(f"{source_name}: les taux infinis ne sont pas admis.")
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError(f"{source_name}: aucun taux numérique exploitable.")
    # Contrôle volontairement large : il détecte une unité restée en %, sans
    # imposer une plage historique trop étroite aux taux négatifs ou extrêmes.
    max_abs = float(np.max(np.abs(finite)))
    if max_abs > 1.0:
        raise ValueError(
            f"{source_name}: |taux|max={max_abs:.6g} après normalisation. "
            "L'unité déclarée est probablement incorrecte; les calculs exigent "
            "des décimales (0.035 = 3.5 %)."
        )
    return normalized


def load_combined_data(
    path: str | Path,
    bam_unit: str = "decimal",
    ecb_unit: str = "percent",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Charge le panel aligné ; suffixes ``_x``=BAM et ``_y``=BCE."""

    df = _read_numeric_csv(path)
    bam_cols = [c for c in df.columns if str(c).endswith("_x")]
    ecb_cols = [c for c in df.columns if str(c).endswith("_y")]
    if not bam_cols or not ecb_cols:
        raise ValueError("Le panel combiné doit contenir des colonnes *_x et *_y.")
    bam = normalize_rate_units(df[bam_cols], bam_unit, "BAM (panel combiné)")
    ecb = normalize_rate_units(df[ecb_cols], ecb_unit, "BCE (panel combiné)")
    _validate_matching_maturities(bam, ecb)
    return bam, ecb


def load_bam_data(path: str | Path, bam_unit: str = "decimal") -> pd.DataFrame:
    """Charge un panel BAM autonome, avec ou sans suffixe ``_x``."""

    df = _read_numeric_csv(path)
    suffixed = [c for c in df.columns if str(c).endswith("_x")]
    bam = df[suffixed].copy() if suffixed else df.copy()
    return normalize_rate_units(bam, bam_unit, "BAM (panel autonome)")


def parse_maturities(columns: Iterable[str]) -> np.ndarray:
    values: list[float] = []
    for raw in columns:
        col = str(raw).strip()
        match = re.match(r"^(\d+(?:\.\d+)?)([MY])(?:_[xy])?$", col, re.I)
        if not match:
            raise ValueError(f"Colonne de maturité non reconnue : {raw}")
        number = float(match.group(1))
        values.append(number / 12.0 if match.group(2).upper() == "M" else number)
    maturities = np.asarray(values, dtype=float)
    if np.any(maturities <= 0) or len(np.unique(maturities)) != len(maturities):
        raise ValueError("Les maturités doivent être positives et uniques.")
    if not np.all(np.diff(maturities) > 0):
        raise ValueError("Les colonnes doivent être rangées par maturité croissante.")
    return maturities


def _validate_matching_maturities(bam: pd.DataFrame, ecb: pd.DataFrame) -> None:
    bam_m = parse_maturities(bam.columns)
    ecb_m = parse_maturities(ecb.columns)
    if len(bam_m) != len(ecb_m) or not np.allclose(bam_m, ecb_m):
        raise ValueError("Les maturités BAM et BCE ne correspondent pas exactement.")


def infer_observed_mask(maturities: np.ndarray, atol: float = 1e-8) -> np.ndarray:
    return np.array(
        [np.any(np.isclose(m, OBSERVED_BAM_MATURITIES, atol=atol)) for m in maturities]
    )


def maturity_weights(maturities: np.ndarray, interpolated_weight: float = 0.35) -> np.ndarray:
    if not 0 < interpolated_weight <= 1:
        raise ValueError("interpolated_weight doit appartenir à ]0, 1].")
    observed = infer_observed_mask(maturities)
    return np.where(observed, 1.0, interpolated_weight)


# ---------------------------------------------------------------------------
# Nelson--Siegel et facteurs transversaux
# ---------------------------------------------------------------------------


def nelson_siegel_loadings(maturities: Sequence[float], decay_time: float) -> np.ndarray:
    """Loadings pour exp(-tau/decay_time), decay_time exprimé en années."""

    tau = np.asarray(maturities, dtype=float)
    if decay_time <= 0 or np.any(tau <= 0):
        raise ValueError("Les maturités et decay_time doivent être strictement positifs.")
    x = tau / decay_time
    slope = -np.expm1(-x) / x
    curvature = slope - np.exp(-x)
    return np.column_stack((np.ones_like(tau), slope, curvature))


def weighted_ols_beta(y: np.ndarray, loadings: np.ndarray, weights: np.ndarray) -> np.ndarray:
    valid = np.isfinite(y) & np.isfinite(weights) & (weights > 0)
    if valid.sum() < 3:
        return np.full(3, np.nan)
    root_w = np.sqrt(weights[valid])
    xw = loadings[valid] * root_w[:, None]
    yw = y[valid] * root_w
    return np.linalg.lstsq(xw, yw, rcond=None)[0]


def extract_ols_betas(
    yields: np.ndarray, loadings: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    return np.vstack([weighted_ols_beta(row, loadings, weights) for row in yields])


def weighted_rmse(error: np.ndarray, weights: np.ndarray) -> float:
    valid = np.isfinite(error)
    w = np.broadcast_to(weights, error.shape)
    valid &= np.isfinite(w) & (w > 0)
    if not valid.any():
        return float("nan")
    return float(np.sqrt(np.sum(w[valid] * error[valid] ** 2) / np.sum(w[valid])))


def weighted_mae(error: np.ndarray, weights: np.ndarray) -> float:
    valid = np.isfinite(error)
    w = np.broadcast_to(weights, error.shape)
    valid &= np.isfinite(w) & (w > 0)
    if not valid.any():
        return float("nan")
    return float(np.sum(w[valid] * np.abs(error[valid])) / np.sum(w[valid]))


# ---------------------------------------------------------------------------
# Dynamique irrégulière, Q et H estimés
# ---------------------------------------------------------------------------


def _project_psd(matrix: np.ndarray, floor: float = 1e-8) -> np.ndarray:
    sym = (matrix + matrix.T) / 2.0
    values, vectors = np.linalg.eigh(sym)
    return vectors @ np.diag(np.maximum(values, floor)) @ vectors.T


def estimate_ou_dynamics(factors: np.ndarray, dates: pd.DatetimeIndex) -> OUParameters:
    """Estime dβ/dt = κ(θ-β)+ε sans supposer un espacement constant."""

    factors = np.asarray(factors, dtype=float)
    valid_rows = np.all(np.isfinite(factors), axis=1)
    factors = factors[valid_rows]
    clean_dates = pd.DatetimeIndex(dates[valid_rows])
    if len(factors) < 30:
        raise ValueError("Au moins 30 courbes valides sont nécessaires à la calibration.")

    dt = np.diff(clean_dates.values).astype("timedelta64[D]").astype(float)
    dt = np.maximum(dt, 1.0)
    previous, current = factors[:-1], factors[1:]
    derivative = (current - previous) / dt[:, None]

    kappa = np.empty(3)
    theta = np.empty(3)
    for j in range(3):
        x = np.column_stack((np.ones(len(previous)), previous[:, j]))
        coef = np.linalg.lstsq(x, derivative[:, j], rcond=None)[0]
        # Une racine explosive est rejetée : le minimum correspond à une quasi-marche aléatoire.
        kappa[j] = np.clip(-coef[1], 1e-5, 0.50)
        raw_theta = coef[0] / kappa[j]
        center = np.nanmedian(factors[:, j])
        scale = max(np.nanstd(factors[:, j]), 1e-3)
        theta[j] = np.clip(raw_theta, center - 8 * scale, center + 8 * scale)

    transition_error = np.empty_like(current)
    for i, delta in enumerate(dt):
        a = np.exp(-kappa * delta)
        mean = a * previous[i] + (1.0 - a) * theta
        transition_error[i] = current[i] - mean
    scaled = transition_error / np.sqrt(dt[:, None])
    q_rate = np.cov(scaled.T, ddof=1)
    return OUParameters(theta, kappa, _project_psd(q_rate, 1e-9))


def transition_matrices(params: OUParameters, delta_days: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    delta = max(float(delta_days), 1.0)
    diagonal = np.exp(-params.kappa * delta)
    a = np.diag(diagonal)
    intercept = (1.0 - diagonal) * params.long_run_mean
    q = _project_psd(params.diffusion_cov * delta, 1e-10)
    return a, intercept, q


def estimate_measurement_cov(
    yields: np.ndarray,
    factors: np.ndarray,
    loadings: np.ndarray,
    variance_floor: float = 1e-6,
) -> np.ndarray:
    residuals = yields - factors @ loadings.T
    variances = np.nanvar(residuals, axis=0, ddof=1)
    fallback = np.nanmedian(variances[np.isfinite(variances) & (variances > 0)])
    if not np.isfinite(fallback):
        fallback = 1e-4
    variances = np.where(np.isfinite(variances), variances, fallback)
    # Diagonale volontaire : robuste lorsque des maturités sont interpolées.
    return np.diag(np.maximum(variances, variance_floor))


def _initial_state(factors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    valid = factors[np.all(np.isfinite(factors), axis=1)]
    if not len(valid):
        raise ValueError("Impossible d'initialiser l'état : aucun facteur OLS valide.")
    mean = valid[0].copy()  # aucune statistique calculée avec des observations futures
    # Prior diffus déterministe : l'ancienne covariance sur les 60 jours suivants
    # contaminait le début de l'échantillon filtré.
    prior_std = np.maximum(np.abs(mean) * 0.25, 0.25)
    cov = np.diag(prior_std**2)
    return mean, _project_psd(cov, 1e-6)


# ---------------------------------------------------------------------------
# Sélection chronologique de la constante de temps
# ---------------------------------------------------------------------------


def select_decay_time(
    yields: np.ndarray,
    dates: pd.DatetimeIndex,
    maturities: np.ndarray,
    weights: np.ndarray,
    grid: np.ndarray | None = None,
    validation_size: int = 252,
    minimum_train: int = 500,
    max_condition: float = 75.0,
) -> tuple[float, pd.DataFrame]:
    """Validation temporelle one-step, entièrement antérieure à l'origine."""

    if grid is None:
        grid = np.geomspace(0.45, 8.0, 28)
    n = len(yields)
    if n < 60:
        raise ValueError("Il faut au moins 60 observations pour choisir decay_time.")
    split = max(minimum_train, n - validation_size)
    split = min(split, n - max(20, min(60, n // 4)))
    diagnostics: list[dict[str, float | bool]] = []

    for decay in np.asarray(grid, dtype=float):
        loadings = nelson_siegel_loadings(maturities, decay)
        cond = float(np.linalg.cond(loadings * np.sqrt(weights)[:, None]))
        peak_years = 1.793 * decay
        admissible = (
            cond <= max_condition
            and peak_years >= maturities.min()
            and peak_years <= min(20.0, maturities.max())
        )
        score = float("inf")
        if admissible:
            factors = extract_ols_betas(yields, loadings, weights)
            try:
                dynamics = estimate_ou_dynamics(factors[:split], dates[:split])
                errors = []
                for t in range(split, n):
                    if not np.all(np.isfinite(factors[t - 1])):
                        continue
                    dt = max((dates[t] - dates[t - 1]).days, 1)
                    a, intercept, _ = transition_matrices(dynamics, dt)
                    beta_pred = a @ factors[t - 1] + intercept
                    errors.append(yields[t] - loadings @ beta_pred)
                if errors:
                    score = weighted_rmse(np.asarray(errors), weights)
            except (ValueError, np.linalg.LinAlgError):
                pass
        diagnostics.append(
            {
                "decay_time": float(decay),
                "validation_rmse": score,
                "condition_number": cond,
                "curvature_peak_years": peak_years,
                "admissible": admissible,
            }
        )

    table = pd.DataFrame(diagnostics)
    finite = table[np.isfinite(table["validation_rmse"])]
    if finite.empty:
        raise RuntimeError("Aucune constante de temps admissible dans la grille.")
    best = float(finite.loc[finite["validation_rmse"].idxmin(), "decay_time"])
    return best, table


def calibrate_dns(
    yields: np.ndarray,
    dates: pd.DatetimeIndex,
    maturities: np.ndarray,
    weights: np.ndarray,
    decay_grid: np.ndarray | None = None,
) -> tuple[DNSModel, np.ndarray, pd.DataFrame]:
    decay, diagnostics = select_decay_time(
        yields, dates, maturities, weights, grid=decay_grid
    )
    loadings = nelson_siegel_loadings(maturities, decay)
    factors = extract_ols_betas(yields, loadings, weights)
    dynamics = estimate_ou_dynamics(factors, dates)
    h = estimate_measurement_cov(yields, factors, loadings)
    init_mean, init_cov = _initial_state(factors)
    model = DNSModel(
        decay,
        loadings,
        dynamics,
        h,
        init_mean,
        init_cov,
        float(np.linalg.cond(loadings * np.sqrt(weights)[:, None])),
    )
    return model, factors, diagnostics


# ---------------------------------------------------------------------------
# BCE exogène et filtre de Kalman stable
# ---------------------------------------------------------------------------


def estimate_ecb_influence(
    bam_factors: np.ndarray,
    ecb_factors: np.ndarray,
    dates: pd.DatetimeIndex,
    bam_dynamics: OUParameters,
    ecb_long_run_mean: np.ndarray,
    window: int = 500,
    ridge: float = 10.0,
) -> np.ndarray:
    """Estime G dans dβ_BAM/dt = κ(θ-β)+G(β_BCE-θ_BCE)+ε."""

    start = max(1, len(bam_factors) - window)
    rows_x, rows_y = [], []
    for t in range(start, len(bam_factors)):
        previous = bam_factors[t - 1]
        current = bam_factors[t]
        ecb_previous = ecb_factors[t - 1]
        if not (
            np.all(np.isfinite(previous))
            and np.all(np.isfinite(current))
            and np.all(np.isfinite(ecb_previous))
        ):
            continue
        dt = max((dates[t] - dates[t - 1]).days, 1)
        derivative = (current - previous) / dt
        baseline = bam_dynamics.kappa * (bam_dynamics.long_run_mean - previous)
        rows_y.append(derivative - baseline)
        rows_x.append(ecb_previous - ecb_long_run_mean)
    if len(rows_x) < 30:
        return np.zeros((3, 3))
    x = np.asarray(rows_x)
    y = np.asarray(rows_y)
    penalty = ridge * np.eye(3)
    # Y = X G' ; résolution Ridge, stable même avec facteurs BCE persistants.
    g_transposed = np.linalg.solve(x.T @ x + penalty, x.T @ y)
    return g_transposed.T


def _safe_cholesky_solve(matrix: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    jitter = 0.0
    for _ in range(7):
        try:
            factor = cho_factor(matrix + np.eye(len(matrix)) * jitter, lower=True)
            return cho_solve(factor, rhs)
        except np.linalg.LinAlgError:
            jitter = 1e-10 if jitter == 0 else jitter * 10
    return np.linalg.solve(matrix + np.eye(len(matrix)) * jitter, rhs)


def kalman_filter(
    yields: np.ndarray,
    dates: pd.DatetimeIndex,
    model: DNSModel,
    ecb_factors: np.ndarray | None = None,
    ecb_influence: np.ndarray | None = None,
    ecb_center: np.ndarray | None = None,
) -> FilterResult:
    """Filtre causal avec forme de Joseph et propagation des jours manquants."""

    n, _ = yields.shape
    state = model.init_mean.copy()
    covariance = model.init_cov.copy()
    filtered = np.full((n, 3), np.nan)
    predicted_next = np.full((n, 3), np.nan)
    covariances = np.full((n, 3, 3), np.nan)
    log_likelihood = 0.0
    identity = np.eye(3)

    for t in range(n):
        if t > 0:
            dt = max((dates[t] - dates[t - 1]).days, 1)
            a, intercept, q = transition_matrices(model.dynamics, dt)
            exogenous = np.zeros(3)
            if ecb_factors is not None and ecb_influence is not None:
                lagged = ecb_factors[t - 1]
                if np.all(np.isfinite(lagged)):
                    center = np.zeros(3) if ecb_center is None else ecb_center
                    exogenous = dt * (ecb_influence @ (lagged - center))
            state = a @ state + intercept + exogenous
            covariance = _project_psd(a @ covariance @ a.T + q, 1e-10)

        obs = yields[t]
        valid = np.isfinite(obs)
        if valid.any():
            loading = model.loadings[valid]
            h_t = model.measurement_cov[np.ix_(valid, valid)]
            innovation = obs[valid] - loading @ state
            innovation_cov = _project_psd(loading @ covariance @ loading.T + h_t, 1e-10)
            solved_innovation = _safe_cholesky_solve(innovation_cov, innovation)
            gain = _safe_cholesky_solve(
                innovation_cov, loading @ covariance
            ).T
            state = state + gain @ innovation
            joseph = identity - gain @ loading
            covariance = _project_psd(
                joseph @ covariance @ joseph.T + gain @ h_t @ gain.T, 1e-10
            )
            sign, logdet = np.linalg.slogdet(innovation_cov)
            if sign > 0:
                log_likelihood -= 0.5 * (
                    valid.sum() * math.log(2 * math.pi)
                    + logdet
                    + innovation @ solved_innovation
                )

        filtered[t] = state
        covariances[t] = covariance

        # β(t+1|t) : BCE(t) est connue à la date t.
        a1, c1, _ = transition_matrices(model.dynamics, 1.0)
        exogenous_next = np.zeros(3)
        if ecb_factors is not None and ecb_influence is not None:
            current_ecb = ecb_factors[t]
            if np.all(np.isfinite(current_ecb)):
                center = np.zeros(3) if ecb_center is None else ecb_center
                exogenous_next = ecb_influence @ (current_ecb - center)
        predicted_next[t] = a1 @ state + c1 + exogenous_next

    return FilterResult(filtered, predicted_next, covariances, float(log_likelihood))


def forecast_states(
    bam_state: np.ndarray,
    bam_model: DNSModel,
    horizon: int,
    ecb_state: np.ndarray | None = None,
    ecb_model: DNSModel | None = None,
    ecb_influence: np.ndarray | None = None,
) -> np.ndarray:
    forecasts = np.empty((horizon, 3))
    state = bam_state.copy()
    current_ecb = None if ecb_state is None else ecb_state.copy()
    a_bam, c_bam, _ = transition_matrices(bam_model.dynamics, 1.0)
    if ecb_model is not None:
        a_ecb, c_ecb, _ = transition_matrices(ecb_model.dynamics, 1.0)

    for h in range(horizon):
        exogenous = np.zeros(3)
        if current_ecb is not None and ecb_influence is not None and ecb_model is not None:
            # Ordre causal : BAM(T+1) utilise BCE(T), puis BCE passe à T+1.
            exogenous = ecb_influence @ (
                current_ecb - ecb_model.dynamics.long_run_mean
            )
        state = a_bam @ state + c_bam + exogenous
        forecasts[h] = state
        if current_ecb is not None and ecb_model is not None:
            current_ecb = a_ecb @ current_ecb + c_ecb
    return forecasts


# ---------------------------------------------------------------------------
# Backtest expanding-window strict
# ---------------------------------------------------------------------------


def _month_origins(dates: pd.DatetimeIndex, start_date: str, horizon: int) -> list[tuple[pd.Timestamp, int]]:
    first = max(pd.Timestamp(start_date), dates[0])
    months = pd.date_range(first.normalize().replace(day=1), dates[-1], freq="MS")
    origins = []
    for month in months:
        cutoff = int(dates.searchsorted(month, side="left") - 1)
        if cutoff >= 60 and cutoff + horizon < len(dates):
            origins.append((month, cutoff))
    return origins


def backtest_walk_forward(
    bam_df: pd.DataFrame,
    ecb_df: pd.DataFrame | None = None,
    start_date: str = "2022-01-01",
    horizon: int = 22,
    interpolated_weight: float = 0.35,
    influence_window: int = 500,
    influence_ridge: float = 10.0,
    decay_grid: np.ndarray | None = None,
) -> list[BacktestOrigin]:
    if ecb_df is not None:
        if not bam_df.index.equals(ecb_df.index):
            raise ValueError("Le backtest enrichi exige des dates BAM/BCE identiques.")
        _validate_matching_maturities(bam_df, ecb_df)
    dates = bam_df.index
    maturities = parse_maturities(bam_df.columns)
    weights = maturity_weights(maturities, interpolated_weight)
    observed = infer_observed_mask(maturities)
    results: list[BacktestOrigin] = []

    for number, (month, cutoff) in enumerate(_month_origins(dates, start_date, horizon), 1):
        train_dates = dates[: cutoff + 1]
        train_bam = bam_df.iloc[: cutoff + 1].to_numpy(float)
        bam_model, bam_ols, _ = calibrate_dns(
            train_bam, train_dates, maturities, weights, decay_grid
        )
        decay_ecb: float | None = None

        if ecb_df is None:
            bam_filter = kalman_filter(train_bam, train_dates, bam_model)
            forecast_betas = forecast_states(bam_filter.filtered[-1], bam_model, horizon)
        else:
            train_ecb = ecb_df.iloc[: cutoff + 1].to_numpy(float)
            ecb_model, ecb_ols, _ = calibrate_dns(
                train_ecb, train_dates, maturities, np.ones_like(weights), decay_grid
            )
            decay_ecb = ecb_model.decay_time
            ecb_filter = kalman_filter(train_ecb, train_dates, ecb_model)
            influence = estimate_ecb_influence(
                bam_ols,
                ecb_filter.filtered,
                train_dates,
                bam_model.dynamics,
                ecb_model.dynamics.long_run_mean,
                influence_window,
                influence_ridge,
            )
            bam_filter = kalman_filter(
                train_bam,
                train_dates,
                bam_model,
                ecb_filter.filtered,
                influence,
                ecb_model.dynamics.long_run_mean,
            )
            forecast_betas = forecast_states(
                bam_filter.filtered[-1],
                bam_model,
                horizon,
                ecb_filter.filtered[-1],
                ecb_model,
                influence,
            )

        future = bam_df.iloc[cutoff + 1 : cutoff + 1 + horizon]
        true_yields = future.to_numpy(float)
        predicted_yields = forecast_betas @ bam_model.loadings.T
        last_curve = bam_df.iloc[cutoff].to_numpy(float)
        persistence = np.repeat(last_curve[None, :], len(future), axis=0)
        error = true_yields - predicted_yields
        persistence_error = true_yields - persistence
        true_betas = extract_ols_betas(true_yields, bam_model.loadings, weights)
        beta_error = true_betas - forecast_betas
        rmse_by_maturity = np.sqrt(np.nanmean(error**2, axis=0))
        observed_error = error[:, observed]
        observed_persistence_error = persistence_error[:, observed]

        results.append(
            BacktestOrigin(
                month,
                dates[cutoff],
                future.index,
                bam_model.decay_time,
                decay_ecb,
                forecast_betas,
                true_betas,
                predicted_yields,
                true_yields,
                persistence,
                weighted_rmse(error, weights),
                weighted_mae(error, weights),
                float(np.sqrt(np.nanmean(observed_error**2))),
                float(np.nanmean(np.abs(observed_error))),
                float(np.sqrt(np.nanmean(observed_persistence_error**2))),
                float(np.nanmean(observed_error**2)),
                float(np.nanmean(observed_persistence_error**2)),
                rmse_by_maturity,
                np.sqrt(np.nanmean(beta_error**2, axis=0)),
            )
        )
        label = "BAM+BCE" if ecb_df is not None else "BAM"
        print(
            f"[{number:02d}] {label} {month:%Y-%m} | "
            f"RMSE observée={results[-1].rmse_observed:.5f} | "
            f"persistance={results[-1].rmse_persistence_observed:.5f}"
        )
    return results


def backtest_table(results: list[BacktestOrigin]) -> pd.DataFrame:
    rows = []
    for result in results:
        row = {
            "rate_unit": "decimal",
            "month": result.month,
            "cutoff_date": result.cutoff_date,
            "last_forecast_date": result.future_dates[-1],
            "n_publications": len(result.future_dates),
            "decay_time_bam": result.decay_time_bam,
            "decay_time_ecb": result.decay_time_ecb,
            "rmse_all_weighted": result.rmse_all,
            "mae_all_weighted": result.mae_all,
            "rmse_observed": result.rmse_observed,
            "mae_observed": result.mae_observed,
            "rmse_persistence_observed": result.rmse_persistence_observed,
        }
        row.update({f"rmse_{FACTOR_NAMES[i]}": result.rmse_beta[i] for i in range(3)})
        rows.append(row)
    return pd.DataFrame(rows)


def diebold_mariano(loss_model: np.ndarray, loss_benchmark: np.ndarray, lag: int = 3) -> dict[str, float]:
    """DM bilatéral avec variance Newey--West ; d<0 favorise le modèle."""

    d = np.asarray(loss_model) - np.asarray(loss_benchmark)
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 8:
        return {"statistic": float("nan"), "p_value": float("nan"), "mean_loss_difference": float("nan")}
    centered = d - d.mean()
    long_run = float(centered @ centered / n)
    for k in range(1, min(lag, n - 1) + 1):
        gamma = float(centered[k:] @ centered[:-k] / n)
        long_run += 2 * (1 - k / (lag + 1)) * gamma
    variance_mean = long_run / n
    if variance_mean <= EPS:
        if abs(d.mean()) <= EPS:
            statistic, p_value = 0.0, 1.0
        else:
            statistic = math.copysign(float("inf"), d.mean())
            p_value = 0.0
    else:
        statistic = float(d.mean() / math.sqrt(variance_mean))
        p_value = float(2 * student_t.sf(abs(statistic), df=n - 1))
    return {"statistic": statistic, "p_value": p_value, "mean_loss_difference": float(d.mean())}


# ---------------------------------------------------------------------------
# Graphiques analytiques en français
# ---------------------------------------------------------------------------


def _save_figure(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_decay_diagnostics(table: pd.DataFrame, best: float, output: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    valid = table[np.isfinite(table["validation_rmse"])]
    axes[0].plot(valid["decay_time"], valid["validation_rmse"] * PLOT_PERCENT, "o-", color="#1f77b4")
    axes[0].axvline(best, color="#d62728", linestyle="--", label=f"retenue={best:.3f}")
    axes[0].set(title="Validation temporelle", xlabel="Constante de temps (années)", ylabel="RMSE one-step (points de %)")
    axes[0].legend()
    axes[1].plot(table["decay_time"], table["condition_number"], "o-", color="#9467bd")
    axes[1].axhline(75, color="black", linestyle="--", label="limite")
    axes[1].set(title="Conditionnement des loadings", xlabel="Constante de temps", ylabel="Nombre de condition")
    axes[1].legend()
    axes[2].plot(table["decay_time"], table["curvature_peak_years"], "o-", color="#ff7f0e")
    axes[2].set(title="Position de la courbure", xlabel="Constante de temps", ylabel="Maturité du pic (années)")
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.suptitle("Sélection chronologique de la constante de temps Nelson–Siegel", fontweight="bold")
    _save_figure(fig, output)


def plot_filter_validation(
    df: pd.DataFrame,
    model: DNSModel,
    result: FilterResult,
    weights: np.ndarray,
    output: Path,
    title: str,
) -> None:
    dates = df.index
    ols = extract_ols_betas(df.to_numpy(float), model.loadings, weights)
    one_step_yields = result.predicted_next[:-1] @ model.loadings.T
    one_step_error = df.to_numpy(float)[1:] - one_step_yields
    fig, axes = plt.subplots(3, 3, figsize=(19, 15))
    for i in range(3):
        axes[0, i].plot(dates, ols[:, i] * PLOT_PERCENT, color=FACTOR_COLORS[i], lw=0.8, label="OLS transversal")
        axes[0, i].plot(dates, result.filtered[:, i] * PLOT_PERCENT, color="black", lw=0.8, alpha=0.8, label="Kalman filtré")
        axes[0, i].set_title(FACTOR_LABELS[i])
        axes[0, i].set_ylabel("Taux (%)")
        axes[0, i].legend(fontsize=8)
        axes[1, i].plot(dates[1:], one_step_error[:, i if i < one_step_error.shape[1] else 0] * PLOT_PERCENT, lw=0.6, color=FACTOR_COLORS[i])
        axes[1, i].axhline(0, color="black", lw=0.8)
        axes[1, i].set_title(f"Erreur one-step — maturité {df.columns[i]}")
        axes[1, i].set_ylabel("Erreur de taux (%)")
    maturities = parse_maturities(df.columns)
    axes[2, 0].plot(maturities, df.iloc[-1] * PLOT_PERCENT, "o-", color="black", label="observée")
    axes[2, 0].plot(maturities, (model.loadings @ result.filtered[-1]) * PLOT_PERCENT, "s--", label="filtrée")
    axes[2, 0].set(title="Dernière courbe", xlabel="Maturité (années)", ylabel="Taux (%)")
    axes[2, 0].legend()
    axes[2, 1].hist(one_step_error[np.isfinite(one_step_error)] * PLOT_PERCENT, bins=60, color="#4c78a8", alpha=0.8)
    axes[2, 1].set(title="Distribution des erreurs sur les taux", xlabel="Erreur (points de %)", ylabel="Fréquence")
    maturity_rmse = np.sqrt(np.nanmean(one_step_error**2, axis=0)) * PLOT_PERCENT
    axes[2, 2].bar(maturities, maturity_rmse, width=np.maximum(maturities * 0.04, 0.1), color="#f58518")
    axes[2, 2].set(title="RMSE one-step par maturité", xlabel="Maturité (années)", ylabel="RMSE (points de %)")
    for ax in axes.ravel():
        ax.grid(alpha=0.22)
    fig.suptitle(title, fontsize=15, fontweight="bold")
    _save_figure(fig, output)


def plot_backtest(results: list[BacktestOrigin], output: Path, title: str) -> None:
    if not results:
        warnings.warn(f"Aucun résultat : graphique {output.name} ignoré.", stacklevel=2)
        return
    months = [r.month for r in results]
    fig, axes = plt.subplots(3, 3, figsize=(19, 15))
    for i in range(3):
        values = [r.rmse_beta[i] * PLOT_PERCENT for r in results]
        axes[0, i].bar(months, values, width=20, color=FACTOR_COLORS[i], alpha=0.8)
        axes[0, i].axhline(np.mean(values), color="black", ls="--", label=f"moyenne={np.mean(values):.4f}")
        axes[0, i].set_title(f"RMSE mensuelle — {FACTOR_LABELS[i]}")
        axes[0, i].set_ylabel("RMSE de taux (%)")
        axes[0, i].legend(fontsize=8)
        last = results[-1]
        axes[1, i].plot(last.future_dates, last.true_betas_ols[:, i] * PLOT_PERCENT, color=FACTOR_COLORS[i], label="OLS futur")
        axes[1, i].plot(last.future_dates, last.predicted_betas[:, i] * PLOT_PERCENT, "k--", label="prévision")
        axes[1, i].set_title(f"Dernière fenêtre — {FACTOR_LABELS[i]}")
        axes[1, i].set_ylabel("Taux (%)")
        axes[1, i].legend(fontsize=8)
        errors = np.concatenate([r.true_betas_ols[:, i] - r.predicted_betas[:, i] for r in results])
        axes[2, i].hist(errors[np.isfinite(errors)] * PLOT_PERCENT, bins=45, color=FACTOR_COLORS[i], alpha=0.75)
        axes[2, i].axvline(0, color="black")
        axes[2, i].set_title(f"Distribution des erreurs — {FACTOR_LABELS[i]}")
        axes[2, i].set_xlabel("Erreur de taux (%)")
    for ax in axes.ravel():
        ax.grid(alpha=0.22)
        if ax.get_subplotspec().is_first_row():
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.suptitle(title, fontsize=15, fontweight="bold")
    _save_figure(fig, output)


def plot_model_comparison(
    base: list[BacktestOrigin], enriched: list[BacktestOrigin], output: Path
) -> None:
    base_by_month = {r.month: r for r in base}
    enriched_by_month = {r.month: r for r in enriched}
    months = sorted(set(base_by_month) & set(enriched_by_month))
    if not months:
        warnings.warn("Aucune fenêtre commune pour la comparaison.", stacklevel=2)
        return
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))
    series = (
        ("DNS–Kalman", [base_by_month[m].rmse_observed * PLOT_PERCENT for m in months], "#1f77b4"),
        ("DNS–Kalman + BCE", [enriched_by_month[m].rmse_observed * PLOT_PERCENT for m in months], "#d62728"),
        ("Persistance", [base_by_month[m].rmse_persistence_observed * PLOT_PERCENT for m in months], "#444444"),
    )
    for label, values, color in series:
        axes[0].plot(months, values, marker="o", ms=3, lw=1.2, label=label, color=color)
    axes[0].set(title="RMSE mensuelle — 9 maturités observées", ylabel="RMSE (points de %)")
    axes[0].legend(fontsize=8)
    means = [np.mean(v) for _, v, _ in series]
    axes[1].bar([s[0] for s in series], means, color=[s[2] for s in series])
    axes[1].set(title="RMSE moyenne", ylabel="RMSE (points de %)")
    axes[1].tick_params(axis="x", rotation=18)
    difference = np.array(series[1][1]) - np.array(series[0][1])
    axes[2].bar(months, difference, width=20, color=np.where(difference < 0, "#2ca02c", "#d62728"))
    axes[2].axhline(0, color="black", lw=0.9)
    axes[2].set(title="BCE − BAM (négatif = amélioration)", ylabel="Écart de RMSE (points de %)")
    for ax in axes:
        ax.grid(alpha=0.22)
        if ax is not axes[1]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.suptitle("Comparaison strictement chronologique des modèles", fontsize=14, fontweight="bold")
    _save_figure(fig, output)


def plot_forecast(
    historical_dates: pd.DatetimeIndex,
    historical_betas: np.ndarray,
    future_dates: pd.DatetimeIndex,
    forecast_betas: np.ndarray,
    last_curve: np.ndarray,
    model: DNSModel,
    maturities: np.ndarray,
    output_dir: Path,
    suffix: str,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for i, ax in enumerate(axes):
        ax.plot(historical_dates[-130:], historical_betas[-130:, i] * PLOT_PERCENT, color=FACTOR_COLORS[i], label="historique filtré")
        ax.plot(future_dates, forecast_betas[:, i] * PLOT_PERCENT, "k--o", ms=3, label="prévision")
        ax.axvline(historical_dates[-1], color="gray", ls=":")
        ax.set_title(FACTOR_LABELS[i])
        ax.set_ylabel("Taux (%)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.22)
    fig.suptitle("Prévision des facteurs à un mois", fontweight="bold")
    _save_figure(fig, output_dir / f"forecast_betas_1month_{suffix}.png")

    forecast_yields = forecast_betas @ model.loadings.T
    fig, ax = plt.subplots(figsize=(12, 6.5))
    ax.plot(maturities, last_curve * PLOT_PERCENT, "o-", color="black", lw=2, label="dernière courbe observée")
    for index, color in zip((4, 9, 21), ("#ff7f0e", "#e377c2", "#9467bd")):
        if index < len(forecast_yields):
            ax.plot(maturities, forecast_yields[index] * PLOT_PERCENT, "^--", color=color, label=f"J+{index + 1}")
    ax.set(title="Projection de la courbe BAM", xlabel="Maturité (années)", ylabel="Taux (%)")
    ax.legend()
    ax.grid(alpha=0.25)
    _save_figure(fig, output_dir / f"forecast_curve_1month_{suffix}.png")


# ---------------------------------------------------------------------------
# Exécution finale, sauvegardes et rapport
# ---------------------------------------------------------------------------


def fit_and_forecast_final(
    bam_df: pd.DataFrame,
    output_dir: Path,
    suffix: str,
    ecb_df: pd.DataFrame | None = None,
    horizon: int = 22,
    interpolated_weight: float = 0.35,
    influence_window: int = 500,
    influence_ridge: float = 10.0,
) -> dict[str, object]:
    dates = bam_df.index
    maturities = parse_maturities(bam_df.columns)
    weights = maturity_weights(maturities, interpolated_weight)
    bam_model, bam_ols, decay_table = calibrate_dns(
        bam_df.to_numpy(float), dates, maturities, weights
    )
    plot_decay_diagnostics(
        decay_table, bam_model.decay_time, output_dir / f"decay_validation_{suffix}.png"
    )
    decay_table.insert(0, "validation_rmse_unit", "decimal")
    decay_table.to_csv(output_dir / f"decay_validation_{suffix}.csv", index=False)

    ecb_model = None
    influence = None
    if ecb_df is None:
        filtered = kalman_filter(bam_df.to_numpy(float), dates, bam_model)
        forecasts = forecast_states(filtered.filtered[-1], bam_model, horizon)
    else:
        ecb_model, _, ecb_decay_table = calibrate_dns(
            ecb_df.to_numpy(float), dates, maturities, np.ones_like(weights)
        )
        ecb_decay_table.insert(0, "validation_rmse_unit", "decimal")
        ecb_decay_table.to_csv(output_dir / "decay_validation_ecb.csv", index=False)
        ecb_filtered = kalman_filter(ecb_df.to_numpy(float), dates, ecb_model)
        influence = estimate_ecb_influence(
            bam_ols,
            ecb_filtered.filtered,
            dates,
            bam_model.dynamics,
            ecb_model.dynamics.long_run_mean,
            influence_window,
            influence_ridge,
        )
        filtered = kalman_filter(
            bam_df.to_numpy(float),
            dates,
            bam_model,
            ecb_filtered.filtered,
            influence,
            ecb_model.dynamics.long_run_mean,
        )
        forecasts = forecast_states(
            filtered.filtered[-1],
            bam_model,
            horizon,
            ecb_filtered.filtered[-1],
            ecb_model,
            influence,
        )

    future_dates = pd.bdate_range(dates[-1] + pd.Timedelta(days=1), periods=horizon)
    beta_history = pd.DataFrame(filtered.filtered, index=dates, columns=FACTOR_NAMES)
    fitted_yields = pd.DataFrame(
        filtered.filtered @ bam_model.loadings.T, index=dates, columns=bam_df.columns
    )
    forecast_beta_df = pd.DataFrame(forecasts, index=future_dates, columns=FACTOR_NAMES)
    forecast_yield_df = pd.DataFrame(
        forecasts @ bam_model.loadings.T, index=future_dates, columns=bam_df.columns
    )
    for frame, filename in (
        (beta_history, f"dns_kalman_betas_{suffix}.csv"),
        (fitted_yields, f"dns_kalman_fitted_yields_{suffix}.csv"),
        (forecast_beta_df, f"forecast_betas_{suffix}.csv"),
        (forecast_yield_df, f"forecast_yields_{suffix}.csv"),
    ):
        saved = frame.copy()
        saved.insert(0, "rate_unit", "decimal")
        saved.to_csv(output_dir / filename)
    plot_filter_validation(
        bam_df,
        bam_model,
        filtered,
        weights,
        output_dir / f"dns_kalman_validation_{suffix}.png",
        f"DNS–Kalman {suffix.upper()} — diagnostic complet",
    )
    plot_forecast(
        dates,
        filtered.filtered,
        future_dates,
        forecasts,
        bam_df.iloc[-1].to_numpy(float),
        bam_model,
        maturities,
        output_dir,
        suffix,
    )
    return {
        "bam_model": bam_model,
        "ecb_model": ecb_model,
        "influence": influence,
        "filter": filtered,
        "forecast_betas": forecast_beta_df,
        "forecast_yields": forecast_yield_df,
    }


def _jsonable_model(model: DNSModel | None) -> dict[str, object] | None:
    if model is None:
        return None
    return {
        "decay_time_years": model.decay_time,
        "condition_number": model.condition_number,
        "long_run_mean": model.dynamics.long_run_mean.tolist(),
        "kappa_per_calendar_day": model.dynamics.kappa.tolist(),
        "diffusion_cov": model.dynamics.diffusion_cov.tolist(),
        "measurement_variances": np.diag(model.measurement_cov).tolist(),
    }


def build_report(
    output_dir: Path,
    base_full: list[BacktestOrigin],
    base_aligned: list[BacktestOrigin],
    enriched: list[BacktestOrigin],
    final_base: dict[str, object],
    final_ecb: dict[str, object],
    used_standalone_bam: bool,
) -> None:
    common_base = {r.month: r for r in base_aligned}
    common_ecb = {r.month: r for r in enriched}
    months = sorted(set(common_base) & set(common_ecb))
    base_losses = np.array([common_base[m].mse_observed for m in months])
    ecb_losses = np.array([common_ecb[m].mse_observed for m in months])
    persistence_losses = np.array([common_base[m].mse_persistence_observed for m in months])
    report = {
        "methodology": {
            "internal_rate_unit": "decimal (0.035 = 3.5 percent)",
            "csv_output_unit": "decimal",
            "plot_unit": "percentage points (internal values multiplied by 100)",
            "backtest": "expanding window; all parameters re-estimated before each origin",
            "horizon": "22 prochaines publications du panel, pas un mois calendaire fixe",
            "decay_convention": "exp(-tau/decay_time), tau and decay_time in years",
            "base_full_bam_calendar": used_standalone_bam,
            "comparison_calendar": "same aligned BAM-ECB dates for both compared models",
            "primary_evaluation": "nine actually observed BAM maturities",
        },
        "windows": {
            "base_full": len(base_full),
            "base_aligned": len(base_aligned),
            "bam_plus_ecb": len(enriched),
            "strictly_common": len(months),
        },
        "mean_rmse_observed": {
            "base_full": float(np.mean([r.rmse_observed for r in base_full])) if base_full else None,
            "base_aligned": float(np.mean([r.rmse_observed for r in base_aligned])) if base_aligned else None,
            "bam_plus_ecb": float(np.mean([r.rmse_observed for r in enriched])) if enriched else None,
            "persistence_common": float(np.mean([common_base[m].rmse_persistence_observed for m in months])) if months else None,
        },
        "diebold_mariano": {
            "base_aligned_vs_persistence": diebold_mariano(base_losses, persistence_losses),
            "bam_plus_ecb_vs_persistence": diebold_mariano(ecb_losses, persistence_losses),
            "bam_plus_ecb_vs_base_aligned": diebold_mariano(ecb_losses, base_losses),
        },
        "final_models": {
            "base": _jsonable_model(final_base["bam_model"]),
            "bam_in_enriched_model": _jsonable_model(final_ecb["bam_model"]),
            "ecb": _jsonable_model(final_ecb["ecb_model"]),
            "ecb_influence": None if final_ecb["influence"] is None else final_ecb["influence"].tolist(),
        },
    }
    (output_dir / "research_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--combined-data", required=True, help="CSV aligné BAM (_x) / BCE (_y)")
    parser.add_argument("--bam-data", help="CSV BAM autonome ; recommandé pour le modèle BAM seul")
    parser.add_argument(
        "--bam-unit",
        choices=VALID_RATE_UNITS,
        default="decimal",
        help="Unité BAM (défaut: decimal; 0.035 représente 3.5 pour cent)",
    )
    parser.add_argument(
        "--ecb-unit",
        choices=VALID_RATE_UNITS,
        default="percent",
        help="Unité BCE (défaut: percent; 3.5 représente 3.5 pour cent)",
    )
    parser.add_argument("--output-dir", default="outputs_corrected")
    parser.add_argument("--start-date", default="2022-01-01")
    parser.add_argument("--horizon", type=int, default=22)
    parser.add_argument("--interpolated-weight", type=float, default=0.35)
    parser.add_argument("--influence-window", type=int, default=500)
    parser.add_argument("--influence-ridge", type=float, default=10.0)
    parser.add_argument("--skip-backtest", action="store_true")
    return parser.parse_args()


def generate_forecasts(
    bam_df: pd.DataFrame,
    output_dir: str | Path,
    suffix: str = "bam",
    ecb_df: pd.DataFrame | None = None,
    **kwargs: object,
) -> dict[str, object]:
    """Public API for final calibration, decimal forecasts, CSVs and figures."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    return fit_and_forecast_final(bam_df, destination, suffix, ecb_df, **kwargs)


def run_backtest(*args: object, **kwargs: object) -> list[BacktestOrigin]:
    """Public API alias for the strict expanding-window backtest."""

    return backtest_walk_forward(*args, **kwargs)


def generate_figures(
    base: list[BacktestOrigin],
    enriched: list[BacktestOrigin],
    output_dir: str | Path,
) -> None:
    """Generate the backtest figures from already-computed decimal results."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    plot_backtest(base, destination / "backtest_bam.png", "Backtest walk-forward — DNS–Kalman BAM")
    plot_backtest(enriched, destination / "backtest_bam_ecb.png", "Backtest walk-forward — DNS–Kalman BAM + BCE")
    plot_model_comparison(base, enriched, destination / "comparison_base_vs_ecb.png")


def run_dns_pipeline(config: PipelineConfig | Mapping[str, object]) -> dict[str, object]:
    """Run the complete research workflow without relying on command-line globals."""

    cfg = config if isinstance(config, PipelineConfig) else PipelineConfig(**config)
    if cfg.horizon < 1:
        raise ValueError("horizon doit être positif.")
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bam_aligned, ecb_aligned = load_combined_data(
        cfg.combined_data, bam_unit=cfg.bam_unit, ecb_unit=cfg.ecb_unit
    )
    if cfg.bam_data:
        bam_full = load_bam_data(cfg.bam_data, bam_unit=cfg.bam_unit)
        parse_maturities(bam_full.columns)
        print("BAM seul : calendrier BAM autonome.")
    else:
        bam_full = bam_aligned
        warnings.warn(
            "--bam-data absent : le modèle BAM seul utilise le calendrier aligné BAM–BCE.",
            stacklevel=2,
        )

    print(
        f"Panel aligné : {len(bam_aligned)} dates, "
        f"{bam_aligned.index.min().date()} → {bam_aligned.index.max().date()}"
    )
    print(
        "Unités normalisées : calculs en décimales; "
        f"BAM source={cfg.bam_unit}, BCE source={cfg.ecb_unit}; "
        "graphiques en %."
    )
    final_base = fit_and_forecast_final(
        bam_full,
        output_dir,
        "bam",
        horizon=cfg.horizon,
        interpolated_weight=cfg.interpolated_weight,
    )
    final_ecb = fit_and_forecast_final(
        bam_aligned,
        output_dir,
        "bam_ecb",
        ecb_aligned,
        cfg.horizon,
        cfg.interpolated_weight,
        cfg.influence_window,
        cfg.influence_ridge,
    )

    if cfg.skip_backtest:
        print("Prévisions produites ; backtest ignoré à la demande.")
        return {"base_final": final_base, "enriched_final": final_ecb}

    # 1) résultat BAM sur toutes ses dates ; 2) base alignée pour comparaison loyale.
    base_full = backtest_walk_forward(
        bam_full,
        start_date=cfg.start_date,
        horizon=cfg.horizon,
        interpolated_weight=cfg.interpolated_weight,
    )
    if bam_full.index.equals(bam_aligned.index) and bam_full.columns.equals(bam_aligned.columns):
        base_aligned = base_full
    else:
        base_aligned = backtest_walk_forward(
            bam_aligned,
            start_date=cfg.start_date,
            horizon=cfg.horizon,
            interpolated_weight=cfg.interpolated_weight,
        )
    enriched = backtest_walk_forward(
        bam_aligned,
        ecb_aligned,
        cfg.start_date,
        cfg.horizon,
        cfg.interpolated_weight,
        cfg.influence_window,
        cfg.influence_ridge,
    )

    backtest_table(base_full).to_csv(output_dir / "backtest_bam_full_calendar.csv", index=False)
    backtest_table(base_aligned).to_csv(output_dir / "backtest_bam_aligned.csv", index=False)
    backtest_table(enriched).to_csv(output_dir / "backtest_bam_ecb.csv", index=False)
    plot_backtest(base_full, output_dir / "backtest_bam.png", "Backtest walk-forward — DNS–Kalman BAM")
    plot_backtest(enriched, output_dir / "backtest_bam_ecb.png", "Backtest walk-forward — DNS–Kalman BAM + BCE")
    plot_model_comparison(base_aligned, enriched, output_dir / "comparison_base_vs_ecb.png")
    build_report(
        output_dir,
        base_full,
        base_aligned,
        enriched,
        final_base,
        final_ecb,
        bool(cfg.bam_data),
    )
    print(f"Pipeline terminé. Résultats : {output_dir.resolve()}")
    return {
        "base_final": final_base,
        "enriched_final": final_ecb,
        "base_backtest": base_full,
        "base_aligned_backtest": base_aligned,
        "enriched_backtest": enriched,
    }


def main() -> None:
    run_dns_pipeline(PipelineConfig(**vars(parse_args())))


if __name__ == "__main__":
    main()
