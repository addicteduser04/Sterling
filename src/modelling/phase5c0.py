"""Leakage-safe simple mean-reversion benchmarks on frozen Phase 5 origins."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from src.modelling import dns
from src.modelling.change_evaluation import dm_hac, overlap_bandwidth
from src.modelling.phase5 import market_curve


MARKETS = {"BAM": "bam_full", "EUROPE": "europe_full", "US_CORE": "us_full"}
NEW_MODELS = ("HISTORICAL_MEAN", "DIRECT_AR1_MEAN_REVERSION",
              "NS_AR1_MEAN_REVERSION", "PCA_AR1_MEAN_REVERSION")


def expanding_historical_mean(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, float)
    if values.ndim != 2 or len(values) == 0:
        raise ValueError("historical mean requires a non-empty 2D training array")
    return np.nanmean(values, axis=0)


def estimate_ar1_mean_reversion(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Estimate stable independent AR(1)s around training-only sample means."""
    values = np.asarray(values, float)
    if values.ndim == 1: values = values[:, None]
    if len(values) < 3: raise ValueError("AR1 estimation requires at least three observations")
    mean = np.nanmean(values, axis=0)
    previous, current = values[:-1] - mean, values[1:] - mean
    denominator = np.sum(previous * previous, axis=0)
    phi = np.divide(np.sum(previous * current, axis=0), denominator,
                    out=np.zeros_like(mean), where=denominator > 0)
    return mean, np.clip(phi, 0.0, 0.999)


def ar1_h_step(current: np.ndarray, mean: np.ndarray, phi: np.ndarray, horizon: int) -> np.ndarray:
    if horizon <= 0: raise ValueError("horizon must be positive")
    return np.asarray(mean) + np.asarray(phi) ** horizon * (np.asarray(current) - np.asarray(mean))


def decimal_to_basis_points(value):
    return np.asarray(value) * 10_000.0


def equilibrium_distance(current: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> float:
    scale = np.maximum(np.asarray(scale, float), 1e-8)
    return float(np.sqrt(np.mean(((np.asarray(current)-np.asarray(mean))/scale)**2)))


def _observed(curve: pd.DataFrame, market: str):
    maturity=dns.parse_maturities(curve.columns)
    mask=np.ones(len(maturity),bool) if market != "BAM" else dns.infer_observed_mask(maturity)
    return curve.loc[:,mask],maturity,mask


def _append(rows, market, origin, target, horizon, maturity, actual, forecast, model):
    for tau,y,yhat in zip(maturity,actual,forecast):
        rows.append({"market":market,"forecast_origin":origin,"target_date":target,"horizon":horizon,
                     "maturity":float(tau),"actual_yield":float(y),"forecast_yield":float(yhat),
                     "error":float(yhat-y),"model":model,"representation":"BENCHMARK",
                     "variant":"training_only_mean_reversion","rate_unit":"decimal"})


def benchmark_market(project: Path, market: str) -> tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame]:
    curve=market_curve(project,market); observed,all_maturity,mask=_observed(curve,market)
    frozen=pd.read_csv(project/"outputs_phase5"/MARKETS[market]/"phase1_forecasts_long.csv",
                       parse_dates=["forecast_origin","target_date"])
    windows=frozen[["forecast_origin","target_date","horizon"]].drop_duplicates().sort_values(["forecast_origin","horizon"])
    dns_meta=frozen[frozen.model.eq("DNS_KALMAN_OU")].groupby("forecast_origin").model_metadata.first()
    rows=[];distances=[];decomposition=[]
    for origin,origin_windows in windows.groupby("forecast_origin"):
        cutoff=curve.index.get_loc(origin); train=curve.iloc[:cutoff+1]; train_obs=observed.iloc[:cutoff+1]
        yield_mean=expanding_historical_mean(train_obs.to_numpy());y_mu,y_phi=estimate_ar1_mean_reversion(train_obs.to_numpy())
        decay=json.loads(dns_meta.loc[origin])["decay_time"]
        loading=dns.nelson_siegel_loadings(all_maturity,decay)
        weights=np.ones(len(all_maturity)) if market!="BAM" else dns.maturity_weights(all_maturity)
        ns=dns.extract_ols_betas(train.to_numpy(),loading,weights);ns_mu,ns_phi=estimate_ar1_mean_reversion(ns)
        pca=PCA(3,svd_solver="full").fit(train_obs);pcs=pca.transform(train_obs);pc_mu,pc_phi=estimate_ar1_mean_reversion(pcs)
        distance=equilibrium_distance(ns[-1],ns_mu,np.std(ns,axis=0,ddof=1))
        category="LOW" if distance<1 else "MEDIUM" if distance<2 else "HIGH"
        for window in origin_windows.itertuples():
            target_pos=curve.index.get_loc(window.target_date);actual=observed.iloc[target_pos].to_numpy()
            forecasts={
                "HISTORICAL_MEAN":yield_mean,
                "DIRECT_AR1_MEAN_REVERSION":ar1_h_step(train_obs.iloc[-1],y_mu,y_phi,window.horizon),
                "NS_AR1_MEAN_REVERSION":(loading@ar1_h_step(ns[-1],ns_mu,ns_phi,window.horizon))[mask],
                "PCA_AR1_MEAN_REVERSION":pca.inverse_transform(ar1_h_step(pcs[-1],pc_mu,pc_phi,window.horizon).reshape(1,-1))[0],
            }
            for model,forecast in forecasts.items():_append(rows,market,origin,window.target_date,window.horizon,
                                                              all_maturity[mask],actual,forecast,model)
            distances.append({"market":market,"forecast_origin":origin,"target_date":window.target_date,
                              "horizon":window.horizon,"standardized_equilibrium_distance":distance,
                              "distance_category":category})
            if market=="BAM" and window.horizon in (66,132,252):
                realised=dns.extract_ols_betas(curve.iloc[[target_pos]].to_numpy(),loading,weights)[0]
                ar_forecast=ar1_h_step(ns[-1],ns_mu,ns_phi,window.horizon)
                for j,name in enumerate(dns.FACTOR_NAMES):
                    decomposition.append({"forecast_origin":origin,"target_date":window.target_date,
                        "horizon":window.horizon,"factor":name,"current_beta":ns[-1,j],"long_run_mean":ns_mu[j],
                        "distance":ns[-1,j]-ns_mu[j],"ar1_phi":ns_phi[j],
                        "mean_reversion_coefficient":ns_phi[j]**window.horizon,
                        "ar1_predicted_beta":ar_forecast[j],"realized_beta":realised[j]})
    return pd.DataFrame(rows),pd.DataFrame(distances),pd.DataFrame(decomposition)


def absolute_metrics(forecasts: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    for key,p in forecasts.groupby(["market","model","horizon"]):
        e=p.error.to_numpy(float);ae=np.abs(e)
        rows.append({"market":key[0],"model":key[1],"horizon":key[2],"n":len(e),
            "rmse_decimal":np.sqrt(np.mean(e*e)),"rmse_percentage_points":100*np.sqrt(np.mean(e*e)),
            "rmse_bp":decimal_to_basis_points(np.sqrt(np.mean(e*e))),"mae_bp":decimal_to_basis_points(np.mean(ae)),
            "median_absolute_error_bp":decimal_to_basis_points(np.median(ae)),"bias_bp":decimal_to_basis_points(np.mean(e)),
            "p75_absolute_error_bp":decimal_to_basis_points(np.quantile(ae,.75)),
            "p90_absolute_error_bp":decimal_to_basis_points(np.quantile(ae,.9)),"maximum_absolute_error_bp":decimal_to_basis_points(ae.max()),
            "within_10bp":np.mean(ae<=.001),"within_25bp":np.mean(ae<=.0025),"within_50bp":np.mean(ae<=.005),"within_100bp":np.mean(ae<=.01)})
    return pd.DataFrame(rows)


def pairwise_dm(forecasts: pd.DataFrame, comparisons) -> pd.DataFrame:
    identity=["market","forecast_origin","target_date","horizon","maturity"];rows=[]
    for model,benchmark in comparisons:
        a=forecasts[forecasts.model.eq(model)][identity+["error"]]
        b=forecasts[forecasts.model.eq(benchmark)][identity+["error"]].rename(columns={"error":"base"})
        for (market,horizon),p in a.merge(b,on=identity).groupby(["market","horizon"]):
            loss=p.assign(a=p.error**2,b=p.base**2).groupby("forecast_origin")[["a","b"]].mean()
            windows=p[["forecast_origin","target_date"]].drop_duplicates();bw=overlap_bandwidth(windows.forecast_origin,windows.target_date)
            rows.append({"market":market,"model":model,"benchmark":benchmark,"horizon":horizon,
                         **dm_hac(np.sqrt(loss.a),np.sqrt(loss.b),horizon,bw)})
    return pd.DataFrame(rows)


def run(project: Path) -> None:
    root=project/"outputs_phase5c0";root.mkdir(exist_ok=True);new=[];dist=[];dec=[]
    frozen=[]
    for market,folder in MARKETS.items():
        f=pd.read_csv(project/"outputs_phase5"/folder/"phase1_forecasts_long.csv",parse_dates=["forecast_origin","target_date"]);f.insert(0,"market",market);frozen.append(f)
        a,b,c=benchmark_market(project,market);new.append(a);dist.append(b);dec.append(c)
    new=pd.concat(new,ignore_index=True);frozen=pd.concat(frozen,ignore_index=True);all_forecasts=pd.concat([frozen,new],ignore_index=True)
    new.to_csv(root/"phase5c0_benchmark_forecasts.csv",index=False);pd.concat(dist).to_csv(root/"phase5c0_equilibrium_distance.csv",index=False);pd.concat(dec).to_csv(root/"phase5c0_bam_factor_decomposition.csv",index=False)
    metrics=absolute_metrics(all_forecasts);metrics.to_csv(root/"phase5c0_absolute_metrics.csv",index=False)
    comparisons=(("DNS_KALMAN_OU","NS_AR1_MEAN_REVERSION"),("DNS_KALMAN_OU","HISTORICAL_MEAN"),("NS_AR1_MEAN_REVERSION","PERSISTENCE"),("PCA_RIDGE_VAR","PCA_AR1_MEAN_REVERSION"),("PCA_AR1_MEAN_REVERSION","PERSISTENCE"))
    pairwise_dm(all_forecasts,comparisons).to_csv(root/"phase5c0_dm_comparisons.csv",index=False)


if __name__=="__main__":run(Path(".").resolve())
