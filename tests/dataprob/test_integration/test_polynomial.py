"""
Fit a five parameter polynomial model to data. 
"""

import pytest

import dataprob

import numpy as np
import matplotlib

def _core_test(method,**method_kwargs):
    
    # ------------------------------------------------------------------------
    # Define model and generate data

    def fourth_order_polynomial(a=1,b=1,c=1,d=1,e=1,x=None): 
          return a + b*x + c*(x**2) + d*(x**3) + e*(x**4)
    
    gen_params = {"a":5,
                "b":0.01,
                "c":0.2,
                "d":0.03,
                "e":0.001}
    
    err = 2
    num_points = 50
    x = np.linspace(-10,10,num_points)
    y_obs = fourth_order_polynomial(x=x,**gen_params) + np.random.normal(loc=0,scale=err,size=num_points)
    y_std = err*2

    # ------------------------------------------------------------------------
    # Define model and generate data

    f = dataprob.setup(fourth_order_polynomial,
                       method=method,
                       non_fit_kwargs={"x":x})
    
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


