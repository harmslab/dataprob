"""
This is a test for bayesian MCMC only. Model generates data uncorrelated with 
input parameter. Our posterior should *only* have information from the prior. 
"""
import pytest

import dataprob

import numpy as np

def _core_test(method,**method_kwargs):
    
    # Thus function returns values that are fixed and uncorrelated with "K". 
    def random_function(K,values): 
        return values
    
    # Create some random data
    gen_params = {"K":1e-3}
    err = 0.05
    num_points = 20
    values = np.random.normal(loc=0,scale=1,size=num_points)
    y_obs = random_function(values=values,**gen_params) 
    y_std = err*2
    
    test_fcn = random_function
    non_fit_kwargs = {"values":values}

    # Set up analysis
    f = dataprob.setup(some_function=test_fcn,
                       method=method,
                       non_fit_kwargs=non_fit_kwargs)

    # Set prior    
    prior_mean = 0
    prior_std = 1   

    f.param_df.loc["K","prior_mean"] = prior_mean
    f.param_df.loc["K","prior_std"] = prior_std

    # Run analysis
    f.fit(y_obs=y_obs,
          y_std=y_std,
          **method_kwargs)

    # Because the data are uncorrelated with model, the posterior better equal
    # the prior!
    posterior_mean = f.fit_df.loc["K","estimate"]
    posterior_std = f.fit_df.loc["K","std"]

    assert np.round(posterior_mean,0) == prior_mean
    assert np.round(posterior_std,0) == prior_std

    # Repeat analysis with same data, different prior. Better recover prior
    # again. 

    # Set up analysis
    f = dataprob.setup(some_function=test_fcn,
                       method=method,
                       non_fit_kwargs=non_fit_kwargs)

    # Set prior    
    prior_mean = 6
    prior_std = 2  

    f.param_df.loc["K","prior_mean"] = prior_mean
    f.param_df.loc["K","prior_std"] = prior_std

    # Run analysis
    f.fit(y_obs=y_obs,
          y_std=y_std,
          **method_kwargs)

    # Because the data are uncorrelated with model, the posterior better equal
    # the prior!
    posterior_mean = f.fit_df.loc["K","estimate"]
    posterior_std = f.fit_df.loc["K","std"]

    assert np.round(posterior_mean,0) == prior_mean
    assert np.round(posterior_std,0) == prior_std



@pytest.mark.slow
def test_bayesian():

    _core_test(method="mcmc",
               use_ml_guess=False,
               num_walkers=100,
               num_steps=200)

