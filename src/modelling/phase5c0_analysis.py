"""Absolute-accuracy and mean-reversion validation diagnostics for Phase 5C0."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import linregress

from src.modelling.phase5c0 import MARKETS, NEW_MODELS, absolute_metrics


PRIMARY=("PERSISTENCE","DNS_KALMAN_OU","NS_RIDGE","NS_ELASTICNET","PCA_RIDGE","PCA_RIDGE_VAR")


def _load(project):
    root=project/"outputs_phase5c0";new=pd.read_csv(root/"phase5c0_benchmark_forecasts.csv",parse_dates=["forecast_origin","target_date"]);old=[]
    for market,folder in MARKETS.items():
        f=pd.read_csv(project/"outputs_phase5"/folder/"phase1_forecasts_long.csv",parse_dates=["forecast_origin","target_date"]);f.insert(0,"market",market);old.append(f)
    return pd.concat([pd.concat(old,ignore_index=True),new],ignore_index=True)


def _calibration(forecasts):
    identity=["market","forecast_origin","target_date","horizon","maturity"]
    current=forecasts[forecasts.model.eq("PERSISTENCE")][identity+["forecast_yield"]].rename(columns={"forecast_yield":"current"})
    x=forecasts.merge(current,on=identity);x["predicted_change"]=x.forecast_yield-x.current;x["realized_change"]=x.actual_yield-x.current
    rows=[]
    for key,p in x.groupby(["market","model","horizon"]):
        pred=p.predicted_change.to_numpy();real=p.realized_change.to_numpy();nonzero=np.sign(pred)!=0
        slope,intercept,r,*_=linregress(pred,real) if np.std(pred)>1e-12 else (np.nan,np.nan,np.nan,np.nan,np.nan)
        rows.append({"market":key[0],"model":key[1],"horizon":key[2],
          "mean_predicted_change_bp":pred.mean()*10000,"mean_realized_change_bp":real.mean()*10000,
          "std_predicted_change_bp":pred.std(ddof=1)*10000,"std_realized_change_bp":real.std(ddof=1)*10000,
          "mean_absolute_predicted_change_bp":np.mean(np.abs(pred))*10000,"mean_absolute_realized_change_bp":np.mean(np.abs(real))*10000,
          "direction_accuracy":np.mean(np.sign(pred[nonzero])==np.sign(real[nonzero])) if nonzero.any() else np.nan,
          "change_correlation":np.corrcoef(pred,real)[0,1] if min(np.std(pred),np.std(real))>1e-12 else np.nan,"calibration_slope_realized_on_predicted":slope})
    return pd.DataFrame(rows),x


def _distance(root,forecasts):
    d=pd.read_csv(root/"phase5c0_equilibrium_distance.csv",parse_dates=["forecast_origin","target_date"])
    f=forecasts[forecasts.market.eq("BAM")&forecasts.model.isin(["PERSISTENCE","DNS_KALMAN_OU"])]
    loss=f.groupby(["forecast_origin","target_date","horizon","model"]).error.apply(lambda z:np.mean(z*z)).unstack()
    loss["dns_gain_bp2"]=(loss.PERSISTENCE-loss.DNS_KALMAN_OU)*1e8
    joined=d[d.market.eq("BAM")].merge(loss[["dns_gain_bp2"]].reset_index(),on=["forecast_origin","target_date","horizon"])
    joined.to_csv(root/"phase5c0_bam_distance_loss.csv",index=False)
    summary=joined.groupby(["horizon","distance_category"]).agg(origins=("forecast_origin","nunique"),mean_distance=("standardized_equilibrium_distance","mean"),mean_dns_loss_gain_bp2=("dns_gain_bp2","mean"),median_dns_loss_gain_bp2=("dns_gain_bp2","median")).reset_index()
    summary.to_csv(root/"phase5c0_distance_quantiles.csv",index=False);return joined


def _bootstrap(root,forecasts):
    rng=np.random.default_rng(42);rows=[]
    specs=(("BAM","DNS_KALMAN_OU","NS_AR1_MEAN_REVERSION"),("EUROPE","PCA_RIDGE_VAR","PCA_AR1_MEAN_REVERSION"),("US_CORE","PCA_RIDGE_VAR","PCA_AR1_MEAN_REVERSION"))
    for market,model,base in specs:
      for h in (44,66,132,252):
        f=forecasts[(forecasts.market.eq(market))&forecasts.horizon.eq(h)&forecasts.model.isin([model,base])]
        z=f.groupby(["forecast_origin","model"]).error.apply(lambda q:np.mean(q*q)).unstack();diff=(z[model]-z[base]).to_numpy();bw={44:2,66:3,132:6,252:12}[h]
        if h>=132:
          rows.append({"market":market,"model":model,"benchmark":base,"horizon":h,"status":"REJECTED_TOO_LITTLE_EFFECTIVE_INFORMATION","block_length":bw+1,"ci_low":np.nan,"ci_high":np.nan});continue
        block=bw+1;starts=np.arange(len(diff)-block+1);means=[]
        for _ in range(2000):
          sample=[]
          while len(sample)<len(diff):sample.extend(diff[rng.choice(starts):][:block])
          means.append(np.mean(sample[:len(diff)]))
        lo,hi=np.quantile(means,[.025,.975]);rows.append({"market":market,"model":model,"benchmark":base,"horizon":h,"status":"MOVING_BLOCK_BOOTSTRAP_DESCRIPTIVE","block_length":block,"ci_low":lo,"ci_high":hi})
    pd.DataFrame(rows).to_csv(root/"phase5c0_block_bootstrap.csv",index=False)


def _plots(root,metrics,forecasts,changes,distance):
    for market,best in (("BAM","DNS_KALMAN_OU"),("EUROPE","PCA_RIDGE_VAR"),("US_CORE","PCA_RIDGE_VAR")):
      p=metrics[(metrics.market.eq(market))&metrics.model.isin(["PERSISTENCE","HISTORICAL_MEAN","DIRECT_AR1_MEAN_REVERSION","NS_AR1_MEAN_REVERSION","PCA_AR1_MEAN_REVERSION",best])]
      fig,ax=plt.subplots(figsize=(8,4.5))
      for model,q in p.groupby("model"):ax.plot(q.horizon,q.rmse_bp,marker="o",label=model)
      ax.set(xscale="log",xticks=[5,10,22,44,66,132,252],title=f"{market}: absolute RMSE",ylabel="RMSE (bp)");ax.get_xaxis().set_major_formatter(plt.ScalarFormatter());ax.legend(fontsize=6,ncol=2);fig.tight_layout();fig.savefig(root/f"fig_{market.lower()}_absolute_rmse.png",dpi=160);plt.close(fig)
    bam=metrics[(metrics.market.eq("BAM"))&metrics.model.eq("DNS_KALMAN_OU")];fig,ax=plt.subplots(figsize=(7,4));ax.plot(bam.horizon,100*(1-bam.set_index('horizon').rmse_bp/metrics[(metrics.market.eq('BAM'))&metrics.model.eq('PERSISTENCE')].set_index('horizon').rmse_bp),marker='o');ax.axhline(0,color='black');ax.set(title='BAM DNS improvement over persistence',xlabel='Horizon',ylabel='RMSE improvement (%)');fig.tight_layout();fig.savefig(root/'fig_bam_dns_improvement.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4.5));
    for h,q in distance.groupby('horizon'):ax.scatter(q.standardized_equilibrium_distance,q.dns_gain_bp2,s=18,alpha=.65,label=f'J+{h}')
    ax.axhline(0,color='black');ax.set(xlabel='Training-only standardized NS distance',ylabel='Persistence loss − DNS loss (bp²)',title='BAM DNS gains versus distance from equilibrium');ax.legend(fontsize=7,ncol=2);fig.tight_layout();fig.savefig(root/'fig_bam_distance_gain.png',dpi=160);plt.close(fig)
    for h in (66,132,252):
      p=changes[(changes.market.eq('BAM'))&changes.model.eq('DNS_KALMAN_OU')&changes.horizon.eq(h)];fig,ax=plt.subplots(figsize=(5,5));ax.scatter(p.realized_change*10000,p.predicted_change*10000,s=12,alpha=.6);lims=np.array([min(ax.get_xlim()[0],ax.get_ylim()[0]),max(ax.get_xlim()[1],ax.get_ylim()[1])]);ax.plot(lims,lims,color='black');ax.set(xlabel='Realized change (bp)',ylabel='Predicted change (bp)',title=f'BAM DNS change calibration J+{h}');fig.tight_layout();fig.savefig(root/f'fig_bam_change_j{h}.png',dpi=160);plt.close(fig)
    important=forecasts[(forecasts.horizon.isin([44,66,132,252]))&(((forecasts.market.eq('BAM'))&forecasts.model.eq('DNS_KALMAN_OU'))|((forecasts.market.ne('BAM'))&forecasts.model.eq('PCA_RIDGE_VAR')))].copy();important['absolute_error_bp']=important.error.abs()*10000;fig,ax=plt.subplots(figsize=(10,5));important.boxplot(column='absolute_error_bp',by=['market','horizon'],ax=ax,showfliers=False,rot=45);fig.suptitle('');ax.set(title='Absolute-error distributions',ylabel='Absolute error (bp)');fig.tight_layout();fig.savefig(root/'fig_error_boxplots.png',dpi=160);plt.close(fig)
    dec=pd.read_csv(root/'phase5c0_bam_representative_decomposition.csv');fig,ax=plt.subplots(figsize=(9,5));x=np.arange(len(dec));ax.scatter(x,dec.current_beta,label='current');ax.scatter(x,dec.long_run_mean,label='mean');ax.scatter(x,dec.dns_predicted_beta,label='DNS forecast');ax.scatter(x,dec.realized_beta,label='realized');ax.set_xticks(x,[f"{r.factor}\n{r.case}" for r in dec.itertuples()],rotation=45);ax.set(title='BAM automatically selected factor decompositions',ylabel='NS factor');ax.legend();fig.tight_layout();fig.savefig(root/'fig_bam_factor_decomposition.png',dpi=160);plt.close(fig)
    growth=pd.read_csv(root/'phase5c0_error_growth.csv');fig,ax=plt.subplots(figsize=(8,4.5))
    for (market,model),p in growth[growth.model.isin(['PERSISTENCE','DNS_KALMAN_OU','PCA_RIDGE_VAR'])].groupby(['market','model']):ax.plot(p.horizon,p.ratio,marker='o',label=f'{market}-{model}')
    ax.set(xscale='log',xticks=[5,10,22,44,66,132,252],title='Forecast-error growth relative to J+5',ylabel='RMSE(h) / RMSE(5)');ax.get_xaxis().set_major_formatter(plt.ScalarFormatter());ax.legend(fontsize=6,ncol=2);fig.tight_layout();fig.savefig(root/'fig_error_growth.png',dpi=160);plt.close(fig)
    half=pd.read_csv(root/'phase5c0_bam_half_life_comparison.csv');fig,ax=plt.subplots(figsize=(7,4.5));x=np.arange(len(half));ax.bar(x-.18,half.ou_half_life_calendar_days,width=.36,label='DNS/OU calendar days');ax.bar(x+.18,half.ar1_half_life_observation_steps,width=.36,label='AR1 observation steps');ax.set_xticks(x,half.factor);ax.set_yscale('log');ax.set(title='BAM factor half-life estimates',ylabel='Half-life (different units; log scale)');ax.legend();fig.tight_layout();fig.savefig(root/'fig_bam_half_life_comparison.png',dpi=160);plt.close(fig)


def run(project:Path):
    root=project/'outputs_phase5c0';forecasts=_load(project);metrics=absolute_metrics(forecasts);metrics.to_csv(root/'phase5c0_absolute_metrics.csv',index=False)
    calibration,changes=_calibration(forecasts);calibration.to_csv(root/'phase5c0_calibration.csv',index=False)
    distance=_distance(root,forecasts);_bootstrap(root,forecasts)
    # Central validation tables.
    original=set(PRIMARY);simple=set(NEW_MODELS)
    rows=[]
    for (market,h),p in metrics.groupby(['market','horizon']):
      rw=p[p.model.eq('PERSISTENCE')].iloc[0];mr=p[p.model.isin(simple)].sort_values('rmse_bp').iloc[0];best=p[p.model.isin(original)].sort_values('rmse_bp').iloc[0]
      rows.append({'market':market,'horizon':h,'persistence_rmse_bp':rw.rmse_bp,'simple_mr_model':mr.model,'simple_mr_rmse_bp':mr.rmse_bp,'best_model':best.model,'best_model_rmse_bp':best.rmse_bp,'best_vs_rw_ratio':best.rmse_bp/rw.rmse_bp,'best_vs_simple_mr_ratio':best.rmse_bp/mr.rmse_bp})
    pd.DataFrame(rows).to_csv(root/'phase5c0_central_table.csv',index=False)
    bam=metrics[(metrics.market.eq('BAM'))&metrics.horizon.isin([22,44,66,132,252])].pivot(index='horizon',columns='model',values='rmse_bp')
    bam['dns_vs_rw_pct']=100*(1-bam.DNS_KALMAN_OU/bam.PERSISTENCE);bam['dns_vs_simple_mr_pct']=100*(1-bam.DNS_KALMAN_OU/bam.NS_AR1_MEAN_REVERSION);bam.reset_index().to_csv(root/'phase5c0_bam_core_table.csv',index=False)
    growth=metrics.assign(ratio=lambda x:x.rmse_bp/x.groupby(['market','model']).rmse_bp.transform('first'));growth.to_csv(root/'phase5c0_error_growth.csv',index=False)
    detailed=[]
    for key,p in forecasts.groupby(['market','model','horizon','maturity']):
      detailed.append({'market':key[0],'model':key[1],'horizon':key[2],'maturity':key[3],'rmse_bp':np.sqrt(np.mean(p.error**2))*10000,'mae_bp':np.mean(np.abs(p.error))*10000})
    pd.DataFrame(detailed).to_csv(root/'phase5c0_maturity_absolute_metrics.csv',index=False)
    # Representative BAM factor cases at J+132, objectively extrema/nearest zero distance.
    dec=pd.read_csv(root/'phase5c0_bam_factor_decomposition.csv');base=dec[dec.horizon.eq(132)];selected=[]
    dnsf=forecasts[(forecasts.market.eq('BAM'))&forecasts.model.eq('DNS_KALMAN_OU')&forecasts.horizon.eq(132)]
    for factor,p in base.groupby('factor'):
      for case,idx in [('large_positive',p.distance.idxmax()),('large_negative',p.distance.idxmin()),('near_equilibrium',p.distance.abs().idxmin())]:
        row=p.loc[idx].copy();maturity=dnsf[dnsf.forecast_origin.eq(pd.Timestamp(row.forecast_origin))];# infer factor from forecast curve using origin lambda
        decay=8.0;loading=__import__('src.modelling.dns',fromlist=['x']).nelson_siegel_loadings(maturity.maturity.to_numpy(),decay);pred=np.linalg.lstsq(loading,maturity.forecast_yield.to_numpy(),rcond=None)[0]
        row['dns_predicted_beta']=pred[list(['beta0','beta1','beta2']).index(factor)];row['case']=case;selected.append(row)
    pd.DataFrame(selected).to_csv(root/'phase5c0_bam_representative_decomposition.csv',index=False)
    # DNS estimates kappa from calendar-day gaps, while its frozen forecast loop
    # advances delta=1 per publication. Keep both units explicit rather than
    # pretending the OU and AR1 half-lives are directly identical.
    unique=dec.drop_duplicates(['forecast_origin','factor']);ar=unique.groupby('factor').ar1_phi.median()
    ou=pd.read_csv(project/'outputs_phase5/phase5_ou_half_lives.csv');ou=ou[ou.market.eq('BAM')].set_index('factor')
    half=[]
    for factor,phi in ar.items():half.append({'factor':factor,'median_ar1_phi':phi,'ar1_half_life_observation_steps':-np.log(2)/np.log(phi) if 0<phi<1 else np.inf,'ou_kappa_per_calendar_day':ou.loc[factor,'kappa_per_day'],'ou_half_life_calendar_days':np.log(2)/ou.loc[factor,'kappa_per_day'],'frozen_dns_transition_increment':'delta=1 per publication step'})
    pd.DataFrame(half).to_csv(root/'phase5c0_bam_half_life_comparison.csv',index=False)
    _plots(root,metrics,forecasts,changes,distance)


if __name__=='__main__':run(Path('.').resolve())
