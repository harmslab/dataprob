"""
Fit a three-parameter saturating exponential model to plausible data. 
"""

import pytest

import dataprob

import numpy as np
import matplotlib

def _core_test(method,**method_kwargs):
    
    # ------------------------------------------------------------------------
    # Define model and generate data

    def exponential_saturation(a,b,k,x): 
        
        return a*(1 - np.exp(-k*(x))) + b

    gen_params = {"a":13,
                  "b":-2,
                  "k":0.5}

    err = 0.3
    num_points = 20

    x = np.linspace(0,10,num_points)
    y_obs = exponential_saturation(x=x,**gen_params) + np.random.normal(0,err,num_points)
    y_std = 2*err

    test_fcn = exponential_saturation
    non_fit_kwargs = {"x":x}

    # ------------------------------------------------------------------------
    # Run analysis

    f = dataprob.setup(some_function=test_fcn,
                    method=method,
                    non_fit_kwargs=non_fit_kwargs)

    f.param_df.loc[["a","b","k"],"guess"] = [1,1,1]

    f.param_df.loc["k","lower_bound"] = 1e-12
    f.param_df.loc["k","upper_bound"] = 2

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

def test_ml():
    
    _core_test(method="ml")

@pytest.mark.slow
def test_bayesian():

    _core_test(method="mcmc")

@pytest.mark.slow
def test_bootstrap():

    _core_test(method="bootstrap")


