"""Phase 6: versioned one-to-five-year BAM mean-reversion forecasts."""

from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd

from src.modelling import dns
from src.modelling.change_evaluation import overlap_bandwidth
from src.modelling.phase4 import DECAY_GRID
from src.modelling.phase4_data import load_bam_phase4_curve
from src.modelling.phase5c0 import ar1_h_step, estimate_ar1_mean_reversion, expanding_historical_mean


HORIZONS=(252,504,756,1008,1260)
MINIMUM_TRAINING_OBSERVATIONS=1260
MODEL_NAMES=("PERSISTENCE","HISTORICAL_MEAN","DIRECT_AR1_MEAN_REVERSION","NS_AR1_MEAN_REVERSION",
             "DNS_CALENDAR_TIME_V2","DNS_PUBLICATION_TIME_V2","DNS_FROZEN_CONVENTION")


def bam_data(project:Path):
    curve=load_bam_phase4_curve(project/'data/combined/bam_ecb_2004.csv');maturity=dns.parse_maturities(curve.columns);mask=dns.infer_observed_mask(maturity)
    return curve,curve.loc[:,mask],maturity,mask


def monthly_windows(index:pd.DatetimeIndex,horizon:int)->pd.DataFrame:
    if len(index)<MINIMUM_TRAINING_OBSERVATIONS:
        return pd.DataFrame(columns=['origin','target','cutoff','calendar_days'])
    first=index[MINIMUM_TRAINING_OBSERVATIONS-1];anchors=pd.date_range(first.normalize().replace(day=1),index[-1],freq='MS');rows=[]
    for anchor in anchors:
        cutoff=int(index.searchsorted(anchor,side='left')-1)
        if cutoff>=MINIMUM_TRAINING_OBSERVATIONS-1 and cutoff+horizon<len(index):
            rows.append({'origin':index[cutoff],'target':index[cutoff+horizon],'cutoff':cutoff,
                         'calendar_days':(index[cutoff+horizon]-index[cutoff]).days})
    return pd.DataFrame(rows)


def horizon_duration(index:pd.DatetimeIndex,horizon:int)->dict:
    days=(index[horizon:]-index[:-horizon]).days.to_numpy()
    return {'horizon':horizon,'median_calendar_days':float(np.median(days)),'p10_calendar_days':float(np.quantile(days,.1)),
            'p90_calendar_days':float(np.quantile(days,.9)),'minimum_calendar_days':int(days.min()),'maximum_calendar_days':int(days.max()),'pairs':len(days)}


def _classify(n,bw,training):
    effective=n/(bw+1) if n else 0
    if not n or training<MINIMUM_TRAINING_OBSERVATIONS:return 'NOT_FEASIBLE'
    if effective>=20:return 'INFERENTIALLY_USABLE'
    if effective>=8:return 'LIMITED_POWER'
    if effective>=4:return 'DESCRIPTIVE_ONLY'
    if effective>=1:return 'SCENARIO_ONLY'
    return 'NOT_FEASIBLE'


def feasibility_audit(project:Path)->tuple[pd.DataFrame,pd.DataFrame]:
    curve,_,_,_=bam_data(project);rows=[];dur=[]
    for h in HORIZONS:
        w=monthly_windows(curve.index,h);bw=overlap_bandwidth(w.origin,w.target) if len(w) else 0;effective=len(w)/(bw+1) if len(w) else 0
        rows.append({'market':'BAM','horizon':h,'available_observations':len(curve),'minimum_training_observations':MINIMUM_TRAINING_OBSERVATIONS,
          'realized_historical_forecasts':len(w),'monthly_forecast_origins':len(w),'non_overlapping_equivalent_windows':round(effective,2),
          'overlap_bandwidth':bw,'effective_inferential_information':round(effective,2),'earliest_possible_origin':w.origin.min().date().isoformat() if len(w) else None,
          'latest_origin_with_realized_target':w.origin.max().date().isoformat() if len(w) else None,
          'remaining_usable_historical_span_days':int((w.origin.max()-w.origin.min()).days) if len(w) else 0,
          'classification':_classify(len(w),bw,MINIMUM_TRAINING_OBSERVATIONS),'classification_basis':'sample structure only; frozen before model results'})
        dur.append(horizon_duration(curve.index,h))
    return pd.DataFrame(rows),pd.DataFrame(dur)


def publication_time_model(yields:np.ndarray,maturities:np.ndarray,weights:np.ndarray,decay:float):
    loading=dns.nelson_siegel_loadings(maturities,decay);factors=dns.extract_ols_betas(yields,loading,weights)
    synthetic=pd.date_range('2000-01-01',periods=len(yields),freq='D')
    dynamics=dns.estimate_ou_dynamics(factors,synthetic);measurement=dns.estimate_measurement_cov(yields,factors,loading);mean,cov=dns._initial_state(factors)
    model=dns.DNSModel(decay,loading,dynamics,measurement,mean,cov,float(np.linalg.cond(loading*np.sqrt(weights)[:,None])))
    filtered=dns.kalman_filter(yields,synthetic,model)
    return model,filtered.filtered[-1],factors


def propagate_ou(state:np.ndarray,model:dns.DNSModel,elapsed:float)->np.ndarray:
    a,c,_=dns.transition_matrices(model.dynamics,elapsed);return a@state+c


def long_run_curve(model:dns.DNSModel)->np.ndarray:
    return model.loadings@model.dynamics.long_run_mean


def _append(rows,origin,target,h,maturity,actual,forecast,model,equilibrium):
    for tau,y,yhat,eq in zip(maturity,actual,forecast,equilibrium):rows.append({'forecast_origin':origin,'target_date':target,'horizon':h,'maturity':float(tau),'actual_yield':float(y),'forecast_yield':float(yhat),'error':float(yhat-y),'model':model,'equilibrium_yield':float(eq),'rate_unit':'decimal','experiment_version':'PHASE6_V2'})


def run_backtest(project:Path,root:Path):
    curve,observed,maturity,mask=bam_data(project);weights=dns.maturity_weights(maturity);windows=pd.concat([monthly_windows(curve.index,h).assign(horizon=h) for h in HORIZONS]).sort_values(['origin','horizon']);rows=[];origin_meta=[]
    for origin,ow in windows.groupby('origin'):
        cutoff=curve.index.get_loc(origin);train=curve.iloc[:cutoff+1];train_obs=observed.iloc[:cutoff+1];y=train.to_numpy()
        calendar_model,calendar_factors,_=dns.calibrate_dns(y,train.index,maturity,weights,DECAY_GRID);calendar_state=dns.kalman_filter(y,train.index,calendar_model).filtered[-1]
        publication_model,publication_state,publication_factors=publication_time_model(y,maturity,weights,calendar_model.decay_time)
        ns_mu,ns_phi=estimate_ar1_mean_reversion(calendar_factors);yield_mu,yield_phi=estimate_ar1_mean_reversion(train_obs.to_numpy());hist=expanding_historical_mean(train_obs.to_numpy());equilibrium=(calendar_model.loadings@ns_mu)[mask]
        for w in ow.itertuples():
            target_pos=curve.index.get_loc(w.target);actual=observed.iloc[target_pos].to_numpy();current=train_obs.iloc[-1].to_numpy()
            forecasts={'PERSISTENCE':current,'HISTORICAL_MEAN':hist,'DIRECT_AR1_MEAN_REVERSION':ar1_h_step(current,yield_mu,yield_phi,w.horizon),
              'NS_AR1_MEAN_REVERSION':(calendar_model.loadings@ar1_h_step(calendar_factors[-1],ns_mu,ns_phi,w.horizon))[mask],
              'DNS_CALENDAR_TIME_V2':(calendar_model.loadings@propagate_ou(calendar_state,calendar_model,w.calendar_days))[mask],
              'DNS_PUBLICATION_TIME_V2':(publication_model.loadings@propagate_ou(publication_state,publication_model,w.horizon))[mask],
              'DNS_FROZEN_CONVENTION':(calendar_model.loadings@propagate_ou(calendar_state,calendar_model,w.horizon))[mask]}
            for model,forecast in forecasts.items():_append(rows,origin,w.target,w.horizon,maturity[mask],actual,forecast,model,equilibrium)
            origin_meta.append({'forecast_origin':origin,'target_date':w.target,'horizon':w.horizon,'calendar_days':w.calendar_days,'decay_time':calendar_model.decay_time,
              'ns_ar1_equilibrium_distance_bp':float(np.sqrt(np.mean((forecasts['NS_AR1_MEAN_REVERSION']-equilibrium)**2))*10000),
              'calendar_dns_equilibrium_distance_bp':float(np.sqrt(np.mean((forecasts['DNS_CALENDAR_TIME_V2']-long_run_curve(calendar_model)[mask])**2))*10000)})
    root.mkdir(exist_ok=True);pd.DataFrame(rows).to_csv(root/'phase6_historical_forecasts.csv',index=False);pd.DataFrame(origin_meta).to_csv(root/'phase6_origin_diagnostics.csv',index=False)


def prospective(project:Path,root:Path,durations:pd.DataFrame):
    curve,observed,maturity,mask=bam_data(project);weights=dns.maturity_weights(maturity);model,factors,_=dns.calibrate_dns(curve.to_numpy(),curve.index,maturity,weights,DECAY_GRID);state=dns.kalman_filter(curve.to_numpy(),curve.index,model).filtered[-1]
    pub,pub_state,_=publication_time_model(curve.to_numpy(),maturity,weights,model.decay_time);mu,phi=estimate_ar1_mean_reversion(factors);equilibrium=(model.loadings@mu)[mask];rows=[]
    for h in HORIZONS:
        days=float(durations.set_index('horizon').loc[h,'median_calendar_days']);forecasts={'NS_AR1_MEAN_REVERSION':(model.loadings@ar1_h_step(factors[-1],mu,phi,h))[mask],
          'DNS_CALENDAR_TIME_V2':(model.loadings@propagate_ou(state,model,days))[mask],'DNS_PUBLICATION_TIME_V2':(pub.loadings@propagate_ou(pub_state,pub,h))[mask]}
        for name,forecast in forecasts.items():
          for tau,current,yhat,eq in zip(maturity[mask],observed.iloc[-1],forecast,equilibrium):rows.append({'forecast_origin':curve.index[-1],'horizon':h,'approximate_target_date':curve.index[-1]+pd.Timedelta(days=days),'maturity':tau,'current_yield':current,'forecast_yield':yhat,'equilibrium_yield':eq,'model':name,'realized_target_available':False})
    pd.DataFrame(rows).to_csv(root/'phase6_latest_prospective_forecasts.csv',index=False)


def run(project:Path):
    root=project/'outputs_phase6';audit,durations=feasibility_audit(project);root.mkdir(exist_ok=True);audit.to_csv(root/'phase6_feasibility_audit.csv',index=False);durations.to_csv(root/'phase6_horizon_calendar_duration.csv',index=False)
    run_backtest(project,root);prospective(project,root,durations)
    (root/'phase6_metadata.json').write_text(json.dumps({'horizons':HORIZONS,'minimum_training_observations':MINIMUM_TRAINING_OBSERVATIONS,'internal_unit':'decimal','origin_frequency':'monthly','historical_phase5_results_modified':False},indent=2)+'\n')


if __name__=='__main__':run(Path('.').resolve())
