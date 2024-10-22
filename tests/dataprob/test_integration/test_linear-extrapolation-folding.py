"""
Fit a six parameter linear extrapolation model often used to analyze 
protein equilibrium unfolding experiments. 
"""

import pytest

import dataprob

import numpy as np
import matplotlib

import warnings

def _core_test(method,**method_kwargs):
    
    # ------------------------------------------------------------------------
    # Define model and generate data

    def linear_extrapolation(dG_unfold=5,m_unfold=-2,
                            b_native=1,m_native=0,
                            b_denat=0,m_denat=0,
                            osmolyte=None,T=298.15,R=0.001987):
        """
        Linear extrapolation unfolding model. 

        Parameters
        ----------
        dG_unfold : float, default=5
            unfolding free energy in water
        m_unfold : float, default=-2
            effect of osmoloyte on the folding energy
        b_native : float, default=1
            intercept of the native baseline
        m_native : float, defualt=0
            slope of the native baseline
        b_denat : float, default=0
            intercept of the denatured baseline
        m_denat : float, defualt=0
            slope of the denatured baseline
        osmolyte : numpy.ndarray
            array of osmolyte concentrations
        T : float, default=298.15
            temperature of experiment in K
        R : float, default=0.001987
            gas constant (default is kcal/mol)

        Returns
        -------
        signal : numpy.ndarray
            protein fraction folded signal as a function of osmolyte
        """
            
        RT = R*T
        dG = dG_unfold + m_unfold*osmolyte
        K = np.exp(-dG/RT)
        
        fx = 1/(1 + K)
        native_signal = (m_native*osmolyte + b_native)*fx
        denatured_signal = (m_denat*osmolyte + b_denat)*(1 - fx)

        return native_signal + denatured_signal
            
    # Parameter for staphylococcal nuclease d+phs protein, pH 7.0
    gen_params = {"dG_unfold":11.9,
                  "m_unfold":-4.2,
                  "b_native":1.5,
                  "m_native":-0.15,
                  "b_denat":0.1,
                  "m_denat":-0.03}

    # Generate data
    T = 298
    R = 0.001987
    err = 0.2
    num_points = 50
    osmolyte = np.linspace(0,8,num_points)

    y_obs_clean = linear_extrapolation(osmolyte=osmolyte,
                                       R=R,T=T,
                                       **gen_params)
    y_obs = y_obs_clean + np.random.normal(0,err,num_points)
    y_std = err*2

    test_fcn = linear_extrapolation
    non_fit_kwargs = {"osmolyte":osmolyte,
                      "R":R,
                      "T":T}

    # ------------------------------------------------------------------------
    # Run analysis

    f = dataprob.setup(some_function=test_fcn,
                       method=method,
                       non_fit_kwargs=non_fit_kwargs)

    # Place bounds to promote relatively rapid convergence of the bayesian MCMC
    # sampler. 
    f.param_df.loc["dG_unfold","lower_bound"] = 0
    f.param_df.loc["dG_unfold","upper_bound"] = 20

    f.param_df.loc["m_unfold","lower_bound"] = -10
    f.param_df.loc["m_unfold","upper_bound"] = 0

    f.param_df.loc["b_native","lower_bound"] = 1.2
    f.param_df.loc["b_native","guess"] = 1.5
    f.param_df.loc["b_native","upper_bound"] = 1.8

    f.param_df.loc["m_native","lower_bound"] = -2
    f.param_df.loc["m_native","guess"] = -1
    f.param_df.loc["m_native","upper_bound"] = 0

    f.param_df.loc["b_denat","lower_bound"] = -0.1
    f.param_df.loc["b_denat","guess"] = 0.1
    f.param_df.loc["b_denat","upper_bound"] = 0.3
    
    f.param_df.loc["m_denat","lower_bound"] = -2
    f.param_df.loc["m_denat","guess"] = -1
    f.param_df.loc["m_denat","upper_bound"] = 0

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
        
        # Catch warnings because a bootstrap replicate often fails and that's 
        # not a big deal. 
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _core_test(method="bootstrap")

    except AssertionError:
        
        # Catch warnings because a bootstrap replicate often fails and that's 
        # not a big deal. 
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _core_test(method="bootstrap")
