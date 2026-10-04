import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from src.modelling import dns
from src.modelling.phase5c0 import (absolute_metrics, ar1_h_step, decimal_to_basis_points,
    equilibrium_distance, estimate_ar1_mean_reversion, expanding_historical_mean)
from src.modelling.phase5c0_analysis import _calibration


def test_expanding_historical_mean_uses_training_rows_only():
    training=np.arange(20,dtype=float).reshape(10,2);future=np.full((2,2),999.)
    expected=expanding_historical_mean(training)
    np.testing.assert_array_equal(expected,expanding_historical_mean(np.vstack([training,future])[:-2]))
    assert not np.allclose(expected,expanding_historical_mean(np.vstack([training,future])))


def test_ar1_estimation_is_history_only_and_stable():
    x=np.arange(80)[:,None]*.001
    mu,phi=estimate_ar1_mean_reversion(x[:60]);mu2,phi2=estimate_ar1_mean_reversion(np.r_[x[:60],[[99.]]][:-1])
    np.testing.assert_array_equal(mu,mu2);np.testing.assert_array_equal(phi,phi2)
    assert 0 <= phi[0] <= .999


def test_closed_form_h_step_ar1():
    np.testing.assert_allclose(ar1_h_step([2.],[1.],[.5],3),[1.125])


def test_factor_specific_ar1_ns_reconstruction():
    factors=np.c_[np.linspace(.02,.03,80),np.linspace(-.01,-.005,80),np.linspace(.002,.004,80)]
    mu,phi=estimate_ar1_mean_reversion(factors);future=ar1_h_step(factors[-1],mu,phi,44)
    loading=dns.nelson_siegel_loadings([.25,1,5,10],2.)
    assert (loading@future).shape==(4,)


def test_direct_yield_ar1_forecast_shape():
    yields=np.random.default_rng(2).normal(.03,.002,(100,8));mu,phi=estimate_ar1_mean_reversion(yields)
    assert ar1_h_step(yields[-1],mu,phi,252).shape==(8,)


def test_pca_ar1_reconstruction_shape():
    yields=np.random.default_rng(3).normal(.03,.002,(100,8));pca=PCA(3).fit(yields);scores=pca.transform(yields)
    mu,phi=estimate_ar1_mean_reversion(scores);future=ar1_h_step(scores[-1],mu,phi,66)
    assert pca.inverse_transform(future.reshape(1,-1)).shape==(1,8)


def test_basis_point_reporting_conversion():
    np.testing.assert_allclose(decimal_to_basis_points([.0001,.01]),[1,100])


def forecast_frame(errors):
    rows=[]
    for i,error in enumerate(errors):rows.append({'market':'X','model':'M','horizon':44,'forecast_origin':pd.Timestamp('2020-01-01')+pd.Timedelta(days=i),'target_date':pd.Timestamp('2020-03-01')+pd.Timedelta(days=i),'maturity':1.,'actual_yield':.03,'forecast_yield':.03+error,'error':error,'representation':'X','variant':'X'})
    return pd.DataFrame(rows)


def test_absolute_error_quantiles_and_bp_metrics():
    m=absolute_metrics(forecast_frame(np.array([.001,.002,.003,.004]))) .iloc[0]
    assert m.median_absolute_error_bp==25
    assert m.p75_absolute_error_bp==32.5 and m.p90_absolute_error_bp==37
    assert m.within_10bp==.25 and m.within_50bp==1


def test_equilibrium_distance_is_standardized_and_training_only():
    assert np.isclose(equilibrium_distance([2,4],[1,2],[1,2]),1)


def test_long_horizon_calibration_metrics():
    base=forecast_frame([-.001,.002]);base['model']='PERSISTENCE';base['forecast_yield']=.03
    model=base.copy();model['model']='AR';model['forecast_yield']=[.031,.029];model['error']=model.forecast_yield-model.actual_yield
    result,_=_calibration(pd.concat([base,model],ignore_index=True))
    row=result[result.model.eq('AR')].iloc[0]
    assert np.isfinite(row.std_predicted_change_bp)


def test_ar1_is_deterministic():
    values=np.random.default_rng(9).normal(size=(100,3));one=estimate_ar1_mean_reversion(values);two=estimate_ar1_mean_reversion(values)
    np.testing.assert_array_equal(one[0],two[0]);np.testing.assert_array_equal(one[1],two[1])
