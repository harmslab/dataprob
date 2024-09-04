"""
Fit a two-parameter model that can describe the pre-saturation growth of 
bacterial cultures. 
"""

import pytest

import dataprob

import numpy as np
import matplotlib

def _core_test(method,**method_kwargs):
    
    # ------------------------------------------------------------------------
    # Define model and generate data

    def lagged_exponential(k=0.1,lag=5,t=None):
        return np.exp(k*(t + lag))

    gen_params = {"k":1/20,
                  "lag":5}

    err = 20
    num_points = 150

    t = np.linspace(0,120,num_points)
    y_obs = lagged_exponential(t=t,**gen_params) + np.random.normal(0,err,num_points)
    y_std = 2*err

    test_fcn = lagged_exponential
    non_fit_kwargs = {"t":t}

    # ------------------------------------------------------------------------
    # Run analysis

    f = dataprob.setup(some_function=test_fcn,
                       method=method,
                       non_fit_kwargs=non_fit_kwargs)

    f.fit(y_obs=y_obs,
          y_std=y_std,
          **method_kwargs)

    # make estimate lands between confidence intervals
    expected = np.array([gen_params[p] for p in f.fit_df.index])
    assert np.sum(expected < np.array(f.fit_df["low_95"])) == 0
    assert np.sum(expected > np.array(f.fit_df["high_95"])) == 0

    fig = dataprob.plot_summary(f)
    assert issubclass(type(fig),matplotlib.figure.Figure)
    matplotlib.pyplot.close(fig)

    fig = dataprob.plot_corner(f)
    assert issubclass(type(fig),matplotlib.figure.Figure)
    matplotlib.pyplot.close(fig)
    
# Try tests twice. We do a lot of tests around 95% confidence intervals. Odds
# are relatively high we hit one across the whole suite. So try once; if fails,
# try again. If it fails again, fail completely

def test_ml():
    
    try:
        _core_test(method="ml")
    except AssertionError:
        _core_test(method="ml")

@pytest.mark.slow
def test_bayesian():

    try:
        _core_test(method="mcmc",max_convergence_cycles=10)
    except AssertionError:
        _core_test(method="mcmc",max_convergence_cycles=10)

@pytest.mark.slow
def test_bootstrap():

    try:
        _core_test(method="bootstrap")
    except AssertionError:
        _core_test(method="bootstrap")