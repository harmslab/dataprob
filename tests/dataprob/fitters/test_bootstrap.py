
import pytest

from dataprob.fitters.bootstrap import BootstrapFitter
from dataprob.model_wrapper.model_wrapper import ModelWrapper

import numpy as np
import pandas as pd


def test_BootstrapFitter__init():

    def test_fcn(a,b): return None

    f = BootstrapFitter(some_function=test_fcn)
    assert f.num_obs is None


def test_BootstrapFitter_fit():

    def test_fcn(m,b,x): return m*x + b

    f = BootstrapFitter(some_function=test_fcn,
                        non_fit_kwargs={"x":np.arange(10)})
    y_obs = np.arange(10)*1 + 2
    y_std = 1.0

    f.fit(y_obs=y_obs,
          y_std=y_std,
          num_bootstrap=3)

    assert f._num_bootstrap == 3

    # This will only warn because the fitter will catch the failure and record
    # it as a failure. It will stick 3 nan values into the samples array
    with pytest.warns():
        f.fit(y_obs=y_obs,
              y_std=y_std,
              num_bootstrap=3,
              not_real_scipy_optimize_kwarg=5)
    assert f.samples.shape == (6,2)
    assert np.sum(np.isnan(f.samples[:3,:])) == 0
    assert np.sum(np.isnan(f.samples[3:,:])) == 6
        
    # Make a new fitter because of wackiness above
    f = BootstrapFitter(some_function=test_fcn,
                        non_fit_kwargs={"x":np.arange(10)})

    with pytest.raises(ValueError):
        f.fit(y_obs=y_obs,
              y_std=y_std,
              num_bootstrap="not_an_integer")
        
    # Need at least two to work
    with pytest.raises(ValueError):
        f.fit(y_obs=y_obs,
              y_std=y_std,
              num_bootstrap=1)
        
    f.fit(y_obs=y_obs,
              y_std=y_std,
              num_bootstrap=2)



def test_BootstrapFitter__fit():
    

    def linear_fcn(m,b,x): return m*x + b
    x = np.linspace(-5,5,10)
    data_df = pd.DataFrame({"y_obs":linear_fcn(m=2,b=-1,x=x),
                            "y_std":0.1*np.ones(10)})

    # -------------------------------------------------------------------------
    # basic run with small number of bootstraps

    f = BootstrapFitter(some_function=linear_fcn,
                        fit_parameters=["m","b"],
                        non_fit_kwargs={"x":x})
    f.data_df = data_df

    assert np.array_equal(f.param_df["guess"],[0,0])
    assert np.sum(np.isnan(f._fit_df["estimate"])) == 2
    assert f.fit_result is None
    assert f.samples is None

    # run containing fit function from base class; that sets fit_has_been_run to
    # true. Make sure containing function ran completely. 
    f.fit(num_bootstrap=10)
    assert f._fit_has_been_run is True

    # These outputs are determined within ._fit
    assert issubclass(type(f._fit_result),dict)
    assert np.array_equal(list(f.fit_result.keys()),["total_samples",
                                                     "num_success",
                                                     "num_failed"])
    assert np.array_equal(f.samples.shape,[10,2]) 
    assert f._success is True
    assert np.sum(np.isnan(f.fit_df["estimate"])) == 0

    # -------------------------------------------------------------------------
    # Run again to make sure bootstraps append properly

    # run containing fit function from base class; that sets fit_has_been_run to
    # true. Make sure containing function ran completely. 
    f.fit(num_bootstrap=10)
    assert f._fit_has_been_run is True

    assert np.array_equal(f.samples.shape,[20,2]) 
    assert f._success is True
    assert np.sum(np.isnan(f.fit_df["estimate"])) == 0

    # -------------------------------------------------------------------------
    # run with a function that always sends out nan -- should warn and show 
    # success == False

    def bad_model(a,b): return np.ones(10)*np.nan
    f = BootstrapFitter(some_function=bad_model)
    f.data_df = data_df
    
    assert np.array_equal(f.param_df["guess"],[0,0])
    assert np.sum(np.isnan(f._fit_df["estimate"])) == 2
    assert f.fit_result is None
    assert f.samples is None
    
    # run containing fit function from base class; that sets fit_has_been_run to
    # true. Make sure containing function ran completely.
    with pytest.warns(): 
        f.fit(num_bootstrap=10)
    assert f._fit_has_been_run is True
    
    # Should not succeed and should not update fit_df
    assert f.success is False
    assert np.sum(np.isnan(f._fit_df["estimate"])) == 2

    # -------------------------------------------------------------------------
    # basic run by set number of steps so small it never converges. should have
    # fit.success == False on each least squares

    f = BootstrapFitter(some_function=linear_fcn,
                        fit_parameters=["m","b"],
                        non_fit_kwargs={"x":x})
    f.data_df = data_df

    assert np.array_equal(f.param_df["guess"],[0,0])
    assert np.sum(np.isnan(f._fit_df["estimate"])) == 2
    assert f.fit_result is None
    assert f.samples is None

    # run containing fit function from base class; that sets fit_has_been_run to
    # true. Make sure containing function ran completely. 
    with pytest.warns():
        f.fit(num_bootstrap=10,
              max_nfev=1)
    assert f._fit_has_been_run is True

    # These outputs are determined within ._fit
    assert np.array_equal(f.samples.shape,[10,2]) 
    assert f._success is False
    assert np.sum(np.isnan(f._fit_df["estimate"])) == 2

    # -------------------------------------------------------------------------
    # Now run fit again without the tiny number of reps --> should now update
    # estimate because enough samples come in. But it should still warn 
    # because lots of samples are nan from last runs. 

    with pytest.warns():
        f.fit(num_bootstrap=10)
    assert f._fit_has_been_run is True

    assert np.array_equal(f.samples.shape,[20,2]) 
    assert f._success is True
    assert np.sum(np.isnan(f.fit_df["estimate"])) == 0

def test_BootstrapFitter__update_fit_df():
    
    # Create a BootstrapFitter with a model loaded (and _fit_df implicitly 
    # created)
    def test_fcn(a=1,b=2): return a*b
    f = BootstrapFitter(some_function=test_fcn)
    
    # add some fake samples
    f._samples = np.random.normal(loc=0,scale=1,size=(10000,2))

    assert np.sum(np.isnan(f._fit_df["estimate"])) == 2
    assert np.sum(np.isnan(f._fit_df["std"])) == 2
    assert np.sum(np.isnan(f._fit_df["low_95"])) == 2
    assert np.sum(np.isnan(f._fit_df["high_95"])) == 2

    f._update_fit_df()

    # Make sure mean/std/95 calc is write given fake samples we stuffed in
    assert np.allclose(np.round(f._fit_df["estimate"],1),[0,0])
    assert np.allclose(np.round(f._fit_df["std"],1),[1,1])
    assert np.allclose(np.round(f._fit_df["low_95"],0),[-2,-2])
    assert np.allclose(np.round(f._fit_df["high_95"],0),[2,2])

    # --------------------------------------------------------------------------
    # Send in np.nan and make sure it handles gracefully -- up to a point

    # Create a BootstrapFitter with a model loaded (and _fit_df implicitly 
    # created)
    def test_fcn(a=1,b=2): return a*b
    f = BootstrapFitter(some_function=test_fcn)
    
    # add some fake samples, then some nans. Should have no effect because we
    # have plenty of samples. 
    f._samples = np.random.normal(loc=0,scale=1,size=(10000,2))
    f._samples[:10,:] = np.nan

    assert np.sum(np.isnan(f._fit_df["estimate"])) == 2
    assert np.sum(np.isnan(f._fit_df["std"])) == 2
    assert np.sum(np.isnan(f._fit_df["low_95"])) == 2
    assert np.sum(np.isnan(f._fit_df["high_95"])) == 2

    f._update_fit_df()

    # Make sure mean/std/95 calc is write given fake samples we stuffed in
    assert np.allclose(np.round(f._fit_df["estimate"],1),[0,0])
    assert np.allclose(np.round(f._fit_df["std"],1),[1,1])
    assert np.allclose(np.round(f._fit_df["low_95"],0),[-2,-2])
    assert np.allclose(np.round(f._fit_df["high_95"],0),[2,2])

    # two non-nan, it should work. Not checking estimate values because there
    # are so few samples
    f._samples[:f._samples.shape[0]-2,:] = np.nan
    assert np.sum(np.isnan(f._samples[:,0])) == 10000 - 2
    f._update_fit_df()

    # only one non-nan sample; should die
    f._samples[:f._samples.shape[0]-1,:] = np.nan
    assert np.sum(np.isnan(f._samples[:,0])) == 10000 - 1
    with pytest.raises(ValueError):
        f._update_fit_df()

    # --------------------------------------------------------------------------
    # make sure the updater properly copies in parameter values the user may 
    # have altered after defining the model but before finalizing the fit. 

    def linear_fcn(m,b,x): return m*x + b
    x = np.linspace(-5,5,10)
    data_df = pd.DataFrame({"y_obs":linear_fcn(m=2,b=-1,x=x),
                            "y_std":0.1*np.ones(10)})
    
    # super small sampler
    f = BootstrapFitter(some_function=linear_fcn,
                        fit_parameters=["m","b"],
                        non_fit_kwargs={"x":x})
    f.data_df = data_df

    # fit_df should have been populated with default values from param_df
    assert np.array_equal(f.fit_df["fixed"],[False,False])
    assert np.array_equal(f.fit_df["guess"],[0,0])
    assert np.array_equal(f.fit_df["prior_mean"],[np.nan,np.nan],equal_nan=True)
    assert np.array_equal(f.fit_df["prior_std"],[np.nan,np.nan],equal_nan=True)
    assert np.array_equal(f.fit_df["lower_bound"],[-np.inf,-np.inf])
    assert np.array_equal(f.fit_df["upper_bound"],[np.inf,np.inf])

    # update param_df
    f.param_df.loc["b","fixed"] = True
    f.param_df.loc["b","guess"] = 1
    f.param_df.loc["b","prior_mean"] = 5
    f.param_df.loc["b","prior_std"] = 3
    f.param_df.loc["m","upper_bound"] = 10
    f.param_df.loc["m","lower_bound"] = -10

    # no change in fit_df yet
    assert np.array_equal(f.fit_df["fixed"],[False,False])
    assert np.array_equal(f.fit_df["guess"],[0,0])
    assert np.array_equal(f.fit_df["prior_mean"],[np.nan,np.nan],equal_nan=True)
    assert np.array_equal(f.fit_df["prior_std"],[np.nan,np.nan],equal_nan=True)
    assert np.array_equal(f.fit_df["lower_bound"],[-np.inf,-np.inf])
    assert np.array_equal(f.fit_df["upper_bound"],[np.inf,np.inf])

    # run containing fit function from base class; that sets fit_has_been_run to
    # true.
    f.fit(num_bootstrap=5)
    assert f._fit_has_been_run is True

    # now fit_df should have been updated with guesses etc. 
    assert np.array_equal(f.fit_df["fixed"],[False,True])
    assert np.array_equal(f.fit_df["guess"],[0,1])
    assert np.array_equal(f.fit_df["prior_mean"],[np.nan,5],equal_nan=True)
    assert np.array_equal(f.fit_df["prior_std"],[np.nan,3],equal_nan=True)
    assert np.array_equal(f.fit_df["lower_bound"],[-10,-np.inf])
    assert np.array_equal(f.fit_df["upper_bound"],[10,np.inf])
    

    # --------------------------------------------------------------------------
    # make sure the function handles a tiny number of samples

    def linear_fcn(m,b,x): return m*x + b
    x = np.linspace(-5,5,10)
    data_df = pd.DataFrame({"y_obs":linear_fcn(m=2,b=-1,x=x),
                            "y_std":0.1*np.ones(10)})
    
    # super small sampler
    f = BootstrapFitter(some_function=linear_fcn,
                        fit_parameters=["m","b"],
                        non_fit_kwargs={"x":x})
    f.data_df = data_df

    assert f.samples is None

    # run containing fit function from base class; that sets fit_has_been_run to
    # true.
    f.fit(num_bootstrap=5)
    assert f._fit_has_been_run is True

    assert f.samples.shape == (5,2)
    f._update_fit_df()


def test_BootstrapFitter___repr__():

    # Stupidly simple fitting problem. find slope
    def model_to_wrap(m=1,x=np.array([1,2,3])): return m*x
    
    # Run _fit_has_been_run, success branch
    f = BootstrapFitter(some_function=model_to_wrap)
    f.fit(y_obs=np.array([2,4,6]),
          y_std=[0.1,0.1,0.1])

    out = f.__repr__().split("\n")
    assert len(out) == 18

    # hack, run _fit_has_been_run, _fit_failed branch
    f._success = False

    out = f.__repr__().split("\n")
    assert len(out) == 13

    # Run not _fit_has_been_run
    f = BootstrapFitter(some_function=model_to_wrap)

    out = f.__repr__().split("\n")
    assert len(out) == 8