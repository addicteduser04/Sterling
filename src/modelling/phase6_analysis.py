"""Phase 6 accuracy, convergence, selection, and presentation figures."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.modelling.change_evaluation import dm_hac, overlap_bandwidth
from src.modelling.phase6 import HORIZONS


def metrics_table(f):
    rows=[]
    current=f[f.model.eq('PERSISTENCE')][['forecast_origin','target_date','horizon','maturity','forecast_yield']].rename(columns={'forecast_yield':'current'})
    x=f.merge(current,on=['forecast_origin','target_date','horizon','maturity'])
    x['predicted_change']=x.forecast_yield-x.current;x['realized_change']=x.actual_yield-x.current
    for key,p in x.groupby(['model','horizon']):
        e=p.error.to_numpy();ae=np.abs(e);pred=p.predicted_change.to_numpy();real=p.realized_change.to_numpy();nz=np.sign(pred)!=0
        rows.append({'model':key[0],'horizon':key[1],'n':len(e),'rmse_bp':np.sqrt(np.mean(e*e))*10000,'mae_bp':np.mean(ae)*10000,
          'median_absolute_error_bp':np.median(ae)*10000,'p75_absolute_error_bp':np.quantile(ae,.75)*10000,'p90_absolute_error_bp':np.quantile(ae,.9)*10000,
          'bias_bp':np.mean(e)*10000,'direction_accuracy':np.mean(np.sign(pred[nz])==np.sign(real[nz])) if nz.any() else np.nan,
          'within_25bp':np.mean(ae<=.0025),'within_50bp':np.mean(ae<=.005),'within_100bp':np.mean(ae<=.01),'within_150bp':np.mean(ae<=.015)})
    return pd.DataFrame(rows),x


def maturity_metrics(f):
    rows=[]
    for key,p in f.groupby(['model','horizon','maturity']):rows.append({'model':key[0],'horizon':key[1],'maturity':key[2],'rmse_bp':np.sqrt(np.mean(p.error**2))*10000,'mae_bp':np.mean(np.abs(p.error))*10000,'bias_bp':np.mean(p.error)*10000})
    return pd.DataFrame(rows)


def dm_guarded(f,audit):
    rows=[];identity=['forecast_origin','target_date','horizon','maturity'];specs=(('DNS_CALENDAR_TIME_V2','NS_AR1_MEAN_REVERSION'),('DNS_PUBLICATION_TIME_V2','NS_AR1_MEAN_REVERSION'),('NS_AR1_MEAN_REVERSION','PERSISTENCE'))
    for model,base in specs:
      a=f[f.model.eq(model)][identity+['error']];b=f[f.model.eq(base)][identity+['error']].rename(columns={'error':'base'})
      for h,p in a.merge(b,on=identity).groupby('horizon'):
        classification=audit.set_index('horizon').loc[h,'classification'];windows=p[['forecast_origin','target_date']].drop_duplicates();bw=overlap_bandwidth(windows.forecast_origin,windows.target_date)
        if classification!='LIMITED_POWER':rows.append({'model':model,'benchmark':base,'horizon':h,'status':'NOT_CALCULATED_INSUFFICIENT_EFFECTIVE_INFORMATION','n':len(windows),'hac_bandwidth':bw,'statistic':np.nan,'p_value':np.nan,'mean_loss_difference':np.nan});continue
        loss=p.assign(a=p.error**2,b=p.base**2).groupby('forecast_origin')[['a','b']].mean();rows.append({'model':model,'benchmark':base,'horizon':h,'status':'LIMITED_POWER_DM',**dm_hac(np.sqrt(loss.a),np.sqrt(loss.b),h,bw)})
    return pd.DataFrame(rows)


def select_examples(f):
    p=f[f.model.eq('NS_AR1_MEAN_REVERSION')];loss=p.groupby(['horizon','forecast_origin']).error.apply(lambda z:np.sqrt(np.mean(z*z))*10000).rename('curve_rmse_bp').reset_index();rows=[]
    for h,q in loss.groupby('horizon'):
      for label,quantile in [('successful_q25',.25),('typical_median',.5),('difficult_q75',.75)]:
        target=q.curve_rmse_bp.quantile(quantile);row=q.iloc[(q.curve_rmse_bp-target).abs().argmin()];rows.append({'horizon':h,'case':label,'forecast_origin':row.forecast_origin,'curve_rmse_bp':row.curve_rmse_bp,'selection_quantile':quantile})
    return pd.DataFrame(rows)


def _curve_figures(root,f,selections):
    maturity=sorted(f.maturity.unique())
    for row in selections.itertuples():
      p=f[(f.horizon.eq(row.horizon))&f.forecast_origin.eq(row.forecast_origin)];base=p[p.model.eq('PERSISTENCE')].sort_values('maturity');model=p[p.model.eq('NS_AR1_MEAN_REVERSION')].sort_values('maturity')
      fig,(ax,err)=plt.subplots(2,1,figsize=(8,6),height_ratios=[3,1],sharex=True)
      ax.plot(maturity,base.forecast_yield*100,marker='o',label='Origin / persistence');ax.plot(maturity,model.forecast_yield*100,marker='o',label='NS-AR1 forecast');ax.plot(maturity,model.actual_yield*100,marker='o',label='Realized');ax.plot(maturity,model.equilibrium_yield*100,ls='--',label='Training-only equilibrium')
      ax.set(ylabel='Yield (%)',title=f'BAM J+{row.horizon} {row.case}: origin {pd.Timestamp(row.forecast_origin).date()}');ax.legend(fontsize=8)
      err.axhline(0,color='black',lw=1);err.bar(maturity,model.error*10000,width=.25);err.set(xlabel='Maturity (years)',ylabel='Error (bp)');fig.tight_layout();fig.savefig(root/f'fig_curve_j{row.horizon}_{row.case}.png',dpi=170);plt.close(fig)


def _summary_figures(root,f,metrics,maturity,origin_diag,prospective,durations):
    models=['PERSISTENCE','HISTORICAL_MEAN','NS_AR1_MEAN_REVERSION','DNS_CALENDAR_TIME_V2','DNS_PUBLICATION_TIME_V2']
    fig,ax=plt.subplots(figsize=(8,4.5))
    for model,p in metrics[metrics.model.isin(models)].groupby('model'):ax.plot(p.horizon/252,p.rmse_bp,marker='o',label=model)
    ax.set(xlabel='Approximate years (h/252)',ylabel='RMSE (bp)',title='BAM absolute forecast accuracy');ax.legend(fontsize=7);fig.tight_layout();fig.savefig(root/'fig_absolute_rmse_horizon.png',dpi=170);plt.close(fig)
    rw=metrics[metrics.model.eq('PERSISTENCE')].set_index('horizon').rmse_bp;fig,ax=plt.subplots(figsize=(8,4.5))
    for model,p in metrics[metrics.model.isin(models[1:])].groupby('model'):ax.plot(p.horizon/252,p.rmse_bp/p.horizon.map(rw),marker='o',label=model)
    ax.axhline(1,color='black');ax.set(xlabel='Approximate years (h/252)',ylabel='RMSE / persistence',title='Relative performance');ax.legend(fontsize=7);fig.tight_layout();fig.savefig(root/'fig_relative_rmse_horizon.png',dpi=170);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4.5))
    for model in ['NS_AR1_MEAN_REVERSION','DNS_CALENDAR_TIME_V2','DNS_PUBLICATION_TIME_V2']:
      p=metrics[metrics.model.eq(model)];ax.plot(p.horizon/252,p.rmse_bp,marker='o',label=model)
    ax.set(xlabel='Approximate years',ylabel='RMSE (bp)',title='Simple NS mean reversion vs corrected DNS');ax.legend();fig.tight_layout();fig.savefig(root/'fig_ns_ar1_vs_dns_v2.png',dpi=170);plt.close(fig)
    sens=f[f.model.isin(['DNS_FROZEN_CONVENTION','DNS_CALENDAR_TIME_V2'])].pivot(index=['forecast_origin','target_date','horizon','maturity'],columns='model',values='forecast_yield').dropna();sens['difference_bp']=(sens.DNS_FROZEN_CONVENTION-sens.DNS_CALENDAR_TIME_V2)*10000;g=sens.groupby('horizon').difference_bp.apply(lambda x:np.sqrt(np.mean(x*x)));fig,ax=plt.subplots(figsize=(7,4));ax.plot(g.index/252,g.values,marker='o');ax.set(xlabel='Approximate years',ylabel='Curve RMS difference (bp)',title='Frozen convention vs calendar-time DNS V2');fig.tight_layout();fig.savefig(root/'fig_dns_time_unit_sensitivity.png',dpi=170);plt.close(fig)
    p=maturity[maturity.model.eq('NS_AR1_MEAN_REVERSION')].pivot(index='maturity',columns='horizon',values='rmse_bp');fig,ax=plt.subplots(figsize=(7,4.5));im=ax.imshow(p,aspect='auto',cmap='viridis');ax.set_xticks(range(len(p.columns)),[f'{h/252:.0f}Y' for h in p.columns]);ax.set_yticks(range(len(p.index)),p.index);ax.set(title='NS-AR1 maturity-specific RMSE',xlabel='Horizon',ylabel='Maturity');fig.colorbar(im,ax=ax,label='RMSE bp');fig.tight_layout();fig.savefig(root/'fig_maturity_error_heatmap.png',dpi=170);plt.close(fig)
    q=origin_diag.groupby('horizon')[['ns_ar1_equilibrium_distance_bp','calendar_dns_equilibrium_distance_bp']].median();fig,ax=plt.subplots(figsize=(7,4));ax.plot(q.index/252,q.ns_ar1_equilibrium_distance_bp,marker='o',label='NS-AR1');ax.plot(q.index/252,q.calendar_dns_equilibrium_distance_bp,marker='o',label='Calendar DNS V2');ax.set(xlabel='Approximate years',ylabel='Median distance to equilibrium (bp)',title='Forecast convergence to equilibrium');ax.legend();fig.tight_layout();fig.savefig(root/'fig_equilibrium_convergence.png',dpi=170);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4));ax.errorbar(durations.horizon/252,durations.median_calendar_days,yerr=[durations.median_calendar_days-durations.p10_calendar_days,durations.p90_calendar_days-durations.median_calendar_days],fmt='o-');ax.set(xlabel='Nominal h/252 years',ylabel='Calendar days',title='Observation-step horizon calendar mapping');fig.tight_layout();fig.savefig(root/'fig_horizon_calendar_duration.png',dpi=170);plt.close(fig)
    # Prospective curve family and maturity evolution, explicitly unrealized.
    pp=prospective[prospective.model.eq('NS_AR1_MEAN_REVERSION')];latest=pp.forecast_origin.iloc[0];fig,ax=plt.subplots(figsize=(8,5));current=pp[pp.horizon.eq(252)].sort_values('maturity');ax.plot(current.maturity,current.current_yield*100,marker='o',lw=2,label='Current')
    for h,z in pp.groupby('horizon'):z=z.sort_values('maturity');ax.plot(z.maturity,z.forecast_yield*100,marker='o',label=f'~{h/252:.0f}Y')
    ax.plot(current.maturity,current.equilibrium_yield*100,ls='--',lw=2,label='Equilibrium');ax.set(xlabel='Maturity (years)',ylabel='Yield (%)',title=f'Prospective BAM curves from {pd.Timestamp(latest).date()} — no realized targets');ax.legend(ncol=2);fig.tight_layout();fig.savefig(root/'fig_latest_prospective_curves.png',dpi=170);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,5))
    for tau,z in pp.groupby('maturity'):z=z.sort_values('horizon');ax.plot(z.horizon/252,z.forecast_yield*100,marker='o',label=f'{tau:g}Y')
    ax.set(xlabel='Approximate years ahead',ylabel='Forecast yield (%)',title='Prospective maturity paths toward equilibrium');ax.legend(fontsize=7,ncol=3);fig.tight_layout();fig.savefig(root/'fig_latest_maturity_evolution.png',dpi=170);plt.close(fig)


def run(project:Path):
    root=project/'outputs_phase6';f=pd.read_csv(root/'phase6_historical_forecasts.csv',parse_dates=['forecast_origin','target_date']);audit=pd.read_csv(root/'phase6_feasibility_audit.csv');dur=pd.read_csv(root/'phase6_horizon_calendar_duration.csv');pros=pd.read_csv(root/'phase6_latest_prospective_forecasts.csv',parse_dates=['forecast_origin','approximate_target_date']);diag=pd.read_csv(root/'phase6_origin_diagnostics.csv')
    metrics,changes=metrics_table(f);metrics.to_csv(root/'phase6_absolute_metrics.csv',index=False);maturity=maturity_metrics(f);maturity.to_csv(root/'phase6_maturity_metrics.csv',index=False);dm_guarded(f,audit).to_csv(root/'phase6_dm_guarded.csv',index=False)
    selection=select_examples(f);selection.to_csv(root/'phase6_curve_example_selection.csv',index=False)
    # Main table and time-unit sensitivity.
    rows=[]
    for h,p in metrics.groupby('horizon'):
      v=p.set_index('model');eligible=p[p.model.ne('DNS_FROZEN_CONVENTION')];best=eligible.loc[eligible.rmse_bp.idxmin()];rows.append({'horizon':h,'classification':audit.set_index('horizon').loc[h,'classification'],'rw_rmse_bp':v.loc['PERSISTENCE','rmse_bp'],'historical_mean_rmse_bp':v.loc['HISTORICAL_MEAN','rmse_bp'],'ns_ar1_rmse_bp':v.loc['NS_AR1_MEAN_REVERSION','rmse_bp'],'dns_calendar_v2_rmse_bp':v.loc['DNS_CALENDAR_TIME_V2','rmse_bp'],'dns_publication_v2_rmse_bp':v.loc['DNS_PUBLICATION_TIME_V2','rmse_bp'],'best_model':best.model,'best_rmse_bp':best.rmse_bp})
    pd.DataFrame(rows).to_csv(root/'phase6_main_accuracy_table.csv',index=False)
    sens=f[f.model.isin(['DNS_FROZEN_CONVENTION','DNS_CALENDAR_TIME_V2'])].pivot(index=['forecast_origin','target_date','horizon','maturity'],columns='model',values=['forecast_yield','error']);out=[]
    for h,p in sens.groupby(level='horizon'):
      diff=(p[('forecast_yield','DNS_FROZEN_CONVENTION')]-p[('forecast_yield','DNS_CALENDAR_TIME_V2')])*10000;out.append({'horizon':h,'frozen_vs_v2_curve_rms_difference_bp':np.sqrt(np.mean(diff**2)),'frozen_dns_rmse_bp':np.sqrt(np.mean(p[('error','DNS_FROZEN_CONVENTION')]**2))*10000,'calendar_v2_rmse_bp':np.sqrt(np.mean(p[('error','DNS_CALENDAR_TIME_V2')]**2))*10000})
    pd.DataFrame(out).to_csv(root/'phase6_dns_time_unit_sensitivity.csv',index=False)
    # Human-facing prospective tables, percentages.
    table=[]
    for (model,tau),p in pros.groupby(['model','maturity']):
      row={'model':model,'maturity':tau,'current_percent':p.current_yield.iloc[0]*100,'equilibrium_percent':p.equilibrium_yield.iloc[0]*100};row.update({f'{h//252}Y_percent':p[p.horizon.eq(h)].forecast_yield.iloc[0]*100 for h in HORIZONS});table.append(row)
    pd.DataFrame(table).to_csv(root/'phase6_latest_forecast_table_percent.csv',index=False)
    _curve_figures(root,f,selection);_summary_figures(root,f,metrics,maturity,diag,pros,dur)


if __name__=='__main__':run(Path('.').resolve())
