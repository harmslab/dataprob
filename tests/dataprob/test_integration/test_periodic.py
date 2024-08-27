"""
Fit a three-parameter periodic function to data. (Note: this would be better 
analyzed with a Fourier Transform for a real application.) Because of periodicity,
it's not well behaved. To improve fitting, fix the frequency and put bounds on 
the phase. Good test of that functionality, as this will not converge with 
no bounds. 
"""

import pytest

import dataprob

import numpy as np
import matplotlib

def _core_test(method,**method_kwargs):
    
    # ------------------------------------------------------------------------
    # Define model and generate data


    def periodic(amplitude,phase,freq,theta):
        return amplitude*np.sin(freq*theta + phase)

    gen_params = {"amplitude":5,
                  "phase":np.pi/2,
                  "freq":2}

    err = 0.4
    num_points = 50

    theta = np.linspace(0,4*np.pi,num_points)
    y_obs = periodic(theta=theta,**gen_params) + np.random.normal(0,err,num_points)
    y_std = err*2

    f = dataprob.setup(periodic,
                       method=method,
                       non_fit_kwargs={"theta":theta})

    # Set the guesses and bounds. Because of the periodicity, this is not
    # particularly well behaved. Fix frequency at right value. 
    f.param_df.loc["amplitude","guess"] = 1
    f.param_df.loc["phase","guess"] = np.pi/2
    f.param_df.loc["freq","guess"] = 2.0
    f.param_df.loc["freq","fixed"] = True

    f.param_df.loc["freq","lower_bound"] = 1.5
    f.param_df.loc["freq","upper_bound"] = 2.5

    f.param_df.loc["phase","lower_bound"] = np.pi/2.5
    f.param_df.loc["phase","upper_bound"] = np.pi/1.5

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





