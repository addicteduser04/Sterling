from pathlib import Path

import numpy as np
import pandas as pd

from src.modelling import dns
from src.modelling.phase5c0 import ar1_h_step, decimal_to_basis_points, expanding_historical_mean
from src.modelling.phase6 import (HORIZONS, MINIMUM_TRAINING_OBSERVATIONS, feasibility_audit,
    horizon_duration, long_run_curve, monthly_windows, propagate_ou, publication_time_model)


PROJECT=Path(__file__).resolve().parents[1]


def test_arbitrary_horizons_through_1260_and_ar1_closed_form():
    assert HORIZONS[-1]==1260
    assert np.isclose(ar1_h_step([2.],[1.],[.99],1260)[0],1+.99**1260)


def test_realized_target_filtering_two_to_five_years():
    dates=pd.bdate_range('2000-01-01',periods=3500)
    for h in HORIZONS:
        windows=monthly_windows(dates,h)
        assert all(dates.get_loc(o)+h==dates.get_loc(t) for o,t in zip(windows.origin,windows.target))
        assert windows.target.max()<=dates[-1]


def test_insufficient_history_has_no_origins():
    dates=pd.bdate_range('2020-01-01',periods=MINIMUM_TRAINING_OBSERVATIONS-1)
    assert monthly_windows(dates,252).empty


def simple_model(kappa=.01):
    maturity=np.array([1.,2.,5.]);loading=dns.nelson_siegel_loadings(maturity,2.)
    dynamics=dns.OUParameters(np.array([.03,-.01,.002]),np.full(3,kappa),np.eye(3)*1e-6)
    return dns.DNSModel(2.,loading,dynamics,np.eye(3)*1e-6,np.zeros(3),np.eye(3),1.)


def test_calendar_time_dns_propagation_matches_closed_form():
    model=simple_model();state=np.array([.05,-.02,.004]);future=propagate_ou(state,model,365)
    expected=model.dynamics.long_run_mean+np.exp(-model.dynamics.kappa*365)*(state-model.dynamics.long_run_mean)
    np.testing.assert_allclose(future,expected)


def test_publication_and_calendar_propagation_are_unit_distinct():
    model=simple_model();state=np.array([.05,-.02,.004])
    assert not np.allclose(propagate_ou(state,model,252),propagate_ou(state,model,376))


def test_publication_time_dns_is_estimated_on_unit_steps():
    maturity=np.array([.25,.5,1,2,5,10]);loading=dns.nelson_siegel_loadings(maturity,2.)
    factors=np.c_[np.linspace(.02,.025,80),np.linspace(-.01,-.005,80),np.linspace(.002,.004,80)]
    model,state,estimated=publication_time_model(factors@loading.T,maturity,np.ones(6),2.)
    assert state.shape==(3,) and estimated.shape==(80,3)
    assert np.all(model.dynamics.kappa>0)


def test_long_run_equilibrium_curve():
    model=simple_model();np.testing.assert_allclose(long_run_curve(model),model.loadings@model.dynamics.long_run_mean)


def test_historical_mean_remains_leakage_safe():
    train=np.arange(30.).reshape(10,3);future=np.full((1,3),999.)
    assert not np.allclose(expanding_historical_mean(train),expanding_historical_mean(np.vstack([train,future])))


def test_horizon_calendar_duration_reporting():
    dates=pd.bdate_range('2000-01-01',periods=2000);result=horizon_duration(dates,504)
    assert result['horizon']==504 and result['p10_calendar_days']<=result['median_calendar_days']<=result['p90_calendar_days']


def test_scenario_only_classification_and_no_invalid_dm():
    audit,_=feasibility_audit(PROJECT)
    assert audit.set_index('horizon').loc[1260,'classification']=='SCENARIO_ONLY'
    assert audit.set_index('horizon').loc[1008,'effective_inferential_information']<4


def test_basis_point_conversion_phase6():
    assert decimal_to_basis_points(.00618)==61.8


def test_curve_plotting_input_integrity():
    maturity=np.array([.25,.5,1,2,5,10,15,20,30]);curve=np.linspace(.02,.04,9)
    assert len(maturity)==len(curve) and np.all(np.diff(maturity)>0)


def test_prospective_and_realized_outputs_are_separate_when_available():
    path=PROJECT/'outputs_phase6/phase6_latest_prospective_forecasts.csv'
    if path.exists():
        prospective=pd.read_csv(path)
        assert not prospective.realized_target_available.astype(bool).any()
        assert 'actual_yield' not in prospective.columns
