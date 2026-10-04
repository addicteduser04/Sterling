"""Phase 5 horizon-response, inference, factor, and stability analysis."""

from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.modelling import dns
from src.modelling.change_evaluation import dm_hac, dm_vs_persistence, metric_table, overlap_bandwidth
from src.modelling.phase4 import DECAY_GRID
from src.modelling.phase5 import COMMON_END, HORIZONS, feasibility_audit, market_curve


MODELS = ("PERSISTENCE", "DNS_KALMAN_OU", "NS_RIDGE", "NS_ELASTICNET", "PCA_RIDGE", "PCA_RIDGE_VAR")


def cached_common(source: Path, destination: Path, end: str = COMMON_END) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    forecasts = pd.read_csv(source / "phase1_forecasts_long.csv", parse_dates=["forecast_origin", "target_date"])
    forecasts = forecasts[forecasts.target_date.le(pd.Timestamp(end))]
    forecasts.to_csv(destination / "phase1_forecasts_long.csv", index=False)
    metric_table(forecasts).to_csv(destination / "phase1_metrics.csv", index=False)
    dm_vs_persistence(forecasts).to_csv(destination / "phase1_dm_vs_persistence.csv", index=False)
    for name in ("phase1_factor_forecasts_long.csv", "phase1_pca_diagnostics.csv", "phase1_sample_diagnostics.csv"):
        frame = pd.read_csv(source / name, parse_dates=["forecast_origin"])
        frame[frame.forecast_origin.isin(forecasts.forecast_origin.unique())].to_csv(destination / name, index=False)
    shutil.copy2(source / "phase5_overlap_audit.csv", destination / "phase5_overlap_audit.csv")


def _load_metrics(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    paths = {"BAM": root/"bam_full", "EUROPE": root/"europe_full", "US_CORE": root/"us_full",
             "BAM_COMMON": root/"bam_common", "EUROPE_COMMON": root/"europe_common", "US_COMMON": root/"us_common"}
    metrics=[]; dms=[]; forecasts=[]
    for market,path in paths.items():
        m=pd.read_csv(path/"phase1_metrics.csv"); m.insert(0,"market",market); metrics.append(m)
        d=pd.read_csv(path/"phase1_dm_vs_persistence.csv"); d.insert(0,"market",market); dms.append(d)
        f=pd.read_csv(path/"phase1_forecasts_long.csv",parse_dates=["forecast_origin","target_date"]); f.insert(0,"market",market); forecasts.append(f)
    return pd.concat(metrics,ignore_index=True),pd.concat(dms,ignore_index=True),pd.concat(forecasts,ignore_index=True)


def _bh(values: pd.Series) -> np.ndarray:
    p=values.fillna(1).to_numpy(); order=np.argsort(p); ranked=p[order]
    adjusted=np.minimum.accumulate((ranked*len(p)/np.arange(1,len(p)+1))[::-1])[::-1]
    result=np.empty(len(p)); result[order]=np.minimum(adjusted,1); return result


def _secondary_dm(forecasts: pd.DataFrame) -> pd.DataFrame:
    comparisons=(("NS_RIDGE","DNS_KALMAN_OU"),("PCA_RIDGE_VAR","DNS_KALMAN_OU"),("PCA_RIDGE_VAR","NS_RIDGE"))
    rows=[]; identity=["forecast_origin","target_date","horizon","maturity"]
    for market,block in forecasts.groupby("market"):
        for model,benchmark in comparisons:
            a=block[block.model.eq(model)][identity+["error"]]
            b=block[block.model.eq(benchmark)][identity+["error"]].rename(columns={"error":"base"})
            matched=a.merge(b,on=identity)
            for horizon,part in matched.groupby("horizon"):
                loss=part.assign(a=part.error**2,b=part.base**2).groupby("forecast_origin")[["a","b"]].mean()
                windows=part[["forecast_origin","target_date"]].drop_duplicates()
                bw=overlap_bandwidth(windows.forecast_origin,windows.target_date)
                rows.append({"market":market,"model":model,"benchmark":benchmark,"horizon":horizon,
                             **dm_hac(np.sqrt(loss.a),np.sqrt(loss.b),horizon,bw)})
    return pd.DataFrame(rows)


def _segments(market: str, maturity: float) -> str:
    if market.startswith("US"):
        return "short" if maturity <= 1 else "medium" if maturity <= 5 else "long"
    return "short" if maturity <= 2 else "medium" if maturity <= 10 else "long"


def _factor_tables(project: Path, root: Path, forecasts: pd.DataFrame) -> None:
    rows=[]; half=[]
    for market in ("BAM","EUROPE","US_CORE"):
        curve=market_curve(project,market)
        maturity=dns.parse_maturities(curve.columns); weights=np.ones(len(maturity))
        model,factors,_=dns.calibrate_dns(curve.to_numpy(),curve.index,maturity,weights,DECAY_GRID)
        frame=pd.DataFrame(factors,index=curve.index,columns=dns.FACTOR_NAMES)
        for factor,kappa in zip(dns.FACTOR_NAMES,model.dynamics.kappa):
            half_days=np.log(2)/kappa if kappa>0 else np.inf
            half.append({"market":market,"factor":factor,"kappa_per_day":kappa,"half_life_days":half_days,
                         "half_life_observations_approx":half_days*5/7})
            for horizon in HORIZONS:
                change=frame[factor].shift(-horizon)-frame[factor]
                rows.append({"market":market,"factor":factor,"horizon":horizon,
                             "change_variance":change.var(),"change_autocorrelation":change.autocorr(1)})
    factor_diag=pd.DataFrame(rows); half=pd.DataFrame(half)
    factor_diag.to_csv(root/"phase5_factor_change_diagnostics.csv",index=False)
    half.to_csv(root/"phase5_ou_half_lives.csv",index=False)
    factor_forecast=[]
    for market,path in (("BAM",root/"bam_full"),("EUROPE",root/"europe_full"),("US_CORE",root/"us_full")):
        x=pd.read_csv(path/"phase1_factor_forecasts_long.csv")
        q=x.groupby(["model","factor","horizon"]).agg(rmse=("error",lambda z:np.sqrt(np.mean(z*z))),
              sign_accuracy=("predicted_change",lambda z:np.nan)).reset_index()
        # Sign accuracy needs aligned realised changes.
        for key,part in x.groupby(["model","factor","horizon"]):
            mask=(q.model.eq(key[0])&q.factor.eq(key[1])&q.horizon.eq(key[2]))
            actual=part.realized_future_factor-part.current_factor
            q.loc[mask,"sign_accuracy"]=np.mean(np.sign(part.predicted_change)==np.sign(actual))
        q.insert(0,"market",market);factor_forecast.append(q)
    pd.concat(factor_forecast).to_csv(root/"phase5_factor_forecast_metrics.csv",index=False)


def _plots(root: Path, metrics: pd.DataFrame, forecasts: pd.DataFrame) -> None:
    primary=metrics[(metrics.maturity.astype(str)=="aggregate")&metrics.market.isin(["BAM","EUROPE","US_CORE"])]
    for market in ("BAM","EUROPE","US_CORE"):
        x=primary[(primary.market.eq(market))&primary.model.ne("PERSISTENCE")]
        fig,ax=plt.subplots(figsize=(8,4.5))
        for model,p in x.groupby("model"): ax.plot(p.horizon,p.rmse_ratio_vs_persistence,marker="o",label=model)
        ax.axhline(1,color="black",lw=1);ax.set(xscale="log",xticks=HORIZONS,title=f"{market}: RMSE ratio by horizon",ylabel="RMSE / persistence")
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter());ax.legend(fontsize=7,ncol=2);fig.tight_layout();fig.savefig(root/f"fig_{market.lower()}_horizon_response.png",dpi=160);plt.close(fig)
    for model in ("DNS_KALMAN_OU","NS_RIDGE","PCA_RIDGE_VAR"):
        x=primary[primary.model.eq(model)];fig,ax=plt.subplots(figsize=(8,4.5))
        for market,p in x.groupby("market"):ax.plot(p.horizon,p.rmse_ratio_vs_persistence,marker="o",label=market)
        ax.axhline(1,color="black",lw=1);ax.set(xscale="log",xticks=HORIZONS,title=f"{model}: cross-market horizon response",ylabel="RMSE / persistence")
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter());ax.legend();fig.tight_layout();fig.savefig(root/f"fig_{model.lower()}_cross_market.png",dpi=160);plt.close(fig)
    rw=primary[primary.model.eq("PERSISTENCE")];fig,ax=plt.subplots(figsize=(8,4.5))
    for market,p in rw.groupby("market"):ax.plot(p.horizon,p.rmse*10000,marker="o",label=market)
    ax.set(xscale="log",xticks=HORIZONS,title="Persistence error growth",ylabel="RMSE (basis points)");ax.get_xaxis().set_major_formatter(plt.ScalarFormatter());ax.legend();fig.tight_layout();fig.savefig(root/"fig_persistence_rmse.png",dpi=160);plt.close(fig)
    detailed=metrics[metrics.market.isin(["BAM","EUROPE","US_CORE"])&metrics.model.isin(["DNS_KALMAN_OU","NS_RIDGE","PCA_RIDGE_VAR"])&metrics.maturity.ne("aggregate")].copy()
    for (market,model),p in detailed.groupby(["market","model"]):
        matrix=p.pivot(index="maturity",columns="horizon",values="rmse_ratio_vs_persistence").astype(float)
        fig,ax=plt.subplots(figsize=(7,4.5));im=ax.imshow(matrix,aspect="auto",vmin=.8,vmax=1.2,cmap="coolwarm")
        ax.set_xticks(range(len(matrix.columns)),matrix.columns);ax.set_yticks(range(len(matrix.index)),matrix.index)
        ax.set(title=f"{market} {model}: maturity × horizon",xlabel="Horizon",ylabel="Maturity (years)");fig.colorbar(im,ax=ax);fig.tight_layout();fig.savefig(root/f"fig_heatmap_{market.lower()}_{model.lower()}.png",dpi=160);plt.close(fig)
    factor=pd.read_csv(root/"phase5_factor_forecast_metrics.csv");fig,ax=plt.subplots(figsize=(8,4.5))
    for (market,factor_name),p in factor[factor.model.eq("NS_RIDGE")].groupby(["market","factor"]):ax.plot(p.horizon,p.rmse,marker="o",label=f"{market}-{factor_name}")
    ax.set(xscale="log",xticks=HORIZONS,title="NS-Ridge factor forecast error",ylabel="Factor RMSE");ax.get_xaxis().set_major_formatter(plt.ScalarFormatter());ax.legend(fontsize=6,ncol=3);fig.tight_layout();fig.savefig(root/"fig_factor_error_horizon.png",dpi=160);plt.close(fig)
    for market in ("BAM","EUROPE","US_CORE"):
        f=forecasts[(forecasts.market.eq(market))&forecasts.horizon.eq(252)]
        agg=primary[(primary.market.eq(market))&primary.horizon.eq(252)&primary.model.ne("PERSISTENCE")]
        best=agg.loc[agg.rmse.idxmin(),"model"]
        z=f[f.model.isin([best,"PERSISTENCE"])].groupby(["forecast_origin","model"]).error.apply(lambda q:np.mean(q*q)).unstack()
        diff=(z[best]-z.PERSISTENCE).cumsum();fig,ax=plt.subplots(figsize=(8,4.5));diff.plot(ax=ax);ax.axhline(0,color="black");ax.set(title=f"{market}: cumulative J+252 loss, {best} minus persistence",ylabel="Cumulative squared-loss difference");fig.tight_layout();fig.savefig(root/f"fig_{market.lower()}_cumulative_j252.png",dpi=160);plt.close(fig)
    common=metrics[(metrics.maturity.astype(str)=="aggregate")&metrics.market.str.contains("COMMON")&metrics.model.ne("PERSISTENCE")]
    full=primary[primary.model.ne("PERSISTENCE")].copy();full["base_market"]=full.market
    common=common.copy();common["base_market"]=common.market.str.replace("_COMMON","")
    merged=full.merge(common,on=["base_market","model","horizon"],suffixes=("_full","_common"));fig,ax=plt.subplots(figsize=(6,5));ax.scatter(merged.rmse_ratio_vs_persistence_full,merged.rmse_ratio_vs_persistence_common,s=12);lims=[min(ax.get_xlim()[0],ax.get_ylim()[0]),max(ax.get_xlim()[1],ax.get_ylim()[1])];ax.plot(lims,lims,color="black");ax.set(xlabel="Full-history ratio",ylabel="Common-period ratio",title="Full history vs common period");fig.tight_layout();fig.savefig(root/"fig_full_vs_common.png",dpi=160);plt.close(fig)
    half=pd.read_csv(root/"phase5_ou_half_lives.csv");fig,ax=plt.subplots(figsize=(8,4.5))
    for market,p in half.groupby("market"):ax.scatter(p.factor,p.half_life_observations_approx,label=market)
    for h in HORIZONS:ax.axhline(h,color="grey",alpha=.15);ax.set(title="OU factor half-lives and forecast horizons",ylabel="Approximate market observations",yscale="log");ax.legend();fig.tight_layout();fig.savefig(root/"fig_ou_half_lives.png",dpi=160);plt.close(fig)


def run(project: Path) -> None:
    root=project/"outputs_phase5";cached_common(root/"bam_full",root/"bam_common");cached_common(root/"europe_full",root/"europe_common")
    metrics,dm,forecasts=_load_metrics(root)
    metrics.to_csv(root/"phase5_metrics_all.csv",index=False);dm.to_csv(root/"phase5_dm_primary.csv",index=False)
    maturity_dm=dm[dm.maturity.astype(str).ne("aggregate")].copy();maturity_dm["fdr_p_value"]=_bh(maturity_dm.p_value);maturity_dm.to_csv(root/"phase5_dm_maturity_fdr.csv",index=False)
    _secondary_dm(forecasts).to_csv(root/"phase5_dm_secondary.csv",index=False)
    aggregate=metrics[metrics.maturity.astype(str).eq("aggregate")].copy();aggregate.to_csv(root/"phase5_aggregate_metrics.csv",index=False)
    master=aggregate[aggregate.market.isin(["BAM","EUROPE","US_CORE"])].pivot(index=["market","model"],columns="horizon",values="rmse_ratio_vs_persistence").reset_index();master.to_csv(root/"phase5_master_horizon_table.csv",index=False)
    primary_dm=dm[dm.maturity.astype(str).eq("aggregate")][["market","model","horizon","p_value","mean_loss_difference","n","hac_bandwidth"]]
    candidates=aggregate[aggregate.market.isin(["BAM","EUROPE","US_CORE"])].copy();best=candidates.loc[candidates.groupby(["market","horizon"]).rmse.idxmin()].merge(primary_dm,on=["market","model","horizon"],how="left")
    audit=feasibility_audit(project)[["market","horizon","classification"]];best=best.merge(audit,on=["market","horizon"])
    best["evidence"]=np.select([(best.rmse_ratio_vs_persistence<.98)&(best.p_value<.05)&(best.mean_loss_difference<0),best.rmse_ratio_vs_persistence<1,best.rmse_ratio_vs_persistence>1.02],["STRONG_POSITIVE","WEAK_POSITIVE","NEGATIVE"],default="NEUTRAL_OR_INCONCLUSIVE")
    best.to_csv(root/"phase5_best_model_by_horizon.csv",index=False)
    cross=[]
    for h,p in best.groupby("horizon"):
        values={row.market:row.rmse_improvement_pct for row in p.itertuples()};models=set(p.model)
        cross.append({"horizon":h,"BAM_best_gain":values.get("BAM"),"Europe_best_gain":values.get("EUROPE"),"US_best_gain":values.get("US_CORE"),"consistent_winner":next(iter(models)) if len(models)==1 else "NO"})
    pd.DataFrame(cross).to_csv(root/"phase5_cross_market_horizon.csv",index=False)
    detail=metrics[metrics.maturity.astype(str).ne("aggregate")&metrics.market.isin(["BAM","EUROPE","US_CORE"])].copy();detail["segment"]=[_segments(m,float(t)) for m,t in zip(detail.market,detail.maturity)];detail.groupby(["market","model","horizon","segment"]).agg(rmse_ratio_mean=("rmse_ratio_vs_persistence","mean"),improving_maturities=("rmse_ratio_vs_persistence",lambda x:int((x<1).sum())),maturities=("maturity","count")).reset_index().to_csv(root/"phase5_curve_segment_results.csv",index=False)
    _factor_tables(project,root,forecasts);_plots(root,metrics,forecasts)


if __name__=="__main__":run(Path(".").resolve())
