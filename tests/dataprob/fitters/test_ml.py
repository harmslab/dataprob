import pytest

from dataprob.fitters.ml import MLFitter
from dataprob.model_wrapper.model_wrapper import ModelWrapper

import numpy as np
import pandas as pd

def test_MLFitter___init__():

    def test_fcn(a,b): return a*b

    f = MLFitter(some_function=test_fcn)
    assert f.num_obs is None

def test_MLFitter_fit():

    def test_fcn(m,b,x): return m*x + b

    f = MLFitter(some_function=test_fcn,
                 non_fit_kwargs={"x":np.arange(10)})
    y_obs = np.arange(10)*1 + 2
    y_std = 1.0

    f.fit(y_obs=y_obs,
          y_std=y_std,
          num_samples=100)

    assert f._num_samples == 100

    # This tests that the scipy.optimize.least_squares pass is happening 
    # correctly.
    with pytest.raises(TypeError):
        f.fit(y_obs=y_obs,
              y_std=y_std,
              num_samples=100,
              not_real_scipy_optimize_kwarg=5)
        
    with pytest.raises(ValueError):
        f.fit(y_obs=y_obs,
              y_std=y_std,
              num_samples="not_an_integer")



def test_MLFitter__fit(linear_fit):

    # Basic functionality and logic tests. Numerical tests on more interesting
    # fitting problems are below. Tests run through .fit() because that
    # initializes everything then calls ._fit(). See the base-class for that. 

    # --------------------------------------------------------------------------
    # Simple model to test

    df = linear_fit["df"]
    fcn = linear_fit["fcn"]  # def simple_linear(m,b,x): return m*x + b
    coeff = linear_fit["coeff"]
    expected_result = np.array([coeff["m"],coeff["b"]])

    # --------------------------------------------------------------------------
    # Run fit

    f = MLFitter(some_function=fcn,
                 fit_parameters=["m","b"],
                 non_fit_kwargs={"x":df.x})
    f._y_obs = df.y_obs
    f._y_std = df.y_std
    
    assert np.sum(np.isnan(f._fit_df["estimate"])) == 2

    # run containing fit function from base class; that sets fit_has_been_run to
    # true.
    f.fit()
    assert f._fit_has_been_run is True

    # now check outputs set by _fit itself
    assert issubclass(type(f._fit_result),dict)
    assert f._fit_result["success"] == True
    assert f._fit_result["status"] == 1
    assert np.allclose(f._fit_result["x"],expected_result)

    # check success flag
    assert f._success is True

    # check that it is setting the fit_df
    assert np.sum(np.isnan(f._fit_df["estimate"])) == 0

    # --------------------------------------------------------------------------
    # This should also delete samples if we run twice. Make sure this is true.

    # There are no samples till we access the property. Check this. 
    assert not hasattr(f,"_samples")
    preserved_samples = f.samples
    assert hasattr(f,"_samples")

    f.fit()
    assert not hasattr(f,"_samples")
    new_samples = f.samples
    assert hasattr(f,"_samples")
    
    assert preserved_samples is not new_samples

    # --------------------------------------------------------------------------
    # Make sure that parameter fixing is propagating properly

    f = MLFitter(some_function=fcn,
                 fit_parameters=["m","b"],
                 non_fit_kwargs={"x":df.x})
    f._y_obs = df.y_obs
    f._y_std = df.y_std

    f.param_df.loc["b","guess"] = 0
    f.param_df.loc["b","fixed"] = True
    assert np.array_equal(f.param_df["fixed"],[False,True])
    f.fit()


def test_MLFitter__update_fit_df(linear_fit):
    
    df = linear_fit["df"]
    fcn = linear_fit["fcn"]  # def simple_linear(m,b,x): return m*x + b
    coeff = linear_fit["coeff"]
    expected_result = np.array([coeff["m"],coeff["b"]])

    f = MLFitter(some_function=fcn,
                 fit_parameters=["m","b"],
                 non_fit_kwargs={"x":df.x})
    f._y_obs = df.y_obs
    f._y_std = df.y_std

    # run containing fit function from base class; that sets fit_has_been_run to
    # true.
    f.fit()
    assert f._fit_has_been_run is True

    # Make sure the dataframe is being populated. Not really testing numerical
    # values, but making sure column assignments make sense. 
    assert len(f._fit_df) == 2
    assert np.allclose(f._fit_df["estimate"],expected_result)
    assert np.sum(np.isnan(f._fit_df["std"])) == 0
    assert np.sum(np.isnan(f._fit_df["low_95"])) == 0
    assert np.sum(np.isnan(f._fit_df["high_95"])) == 0
    assert np.sum(f._fit_df["low_95"] < f._fit_df["estimate"]) == 2
    assert np.sum(f._fit_df["high_95"] > f._fit_df["estimate"]) == 2
    assert np.sum(f._fit_df["low_95"] < (f._fit_df["estimate"] - f._fit_df["std"])) == 2
    assert np.sum(f._fit_df["high_95"] > (f._fit_df["estimate"] + f._fit_df["std"])) == 2

    # Hack so the jacobian is now a singular matrix. This will cause the 
    # function to throw a warning and set values to nan
    f._fit_result.jac = np.zeros(f._fit_result.jac.shape,dtype=float)
    
    with pytest.warns():
        f._update_fit_df()
    assert np.allclose(f._fit_df["estimate"],expected_result)
    assert np.sum(np.isnan(f._fit_df["std"])) == 2
    assert np.sum(np.isnan(f._fit_df["low_95"])) == 2
    assert np.sum(np.isnan(f._fit_df["high_95"])) == 2
    
    # --------------------------------------------------------------------------
    # make sure the updater properly copies in parameter values the user may 
    # have altered after defining the model but before finalizing the fit. 

    f = MLFitter(some_function=fcn,
                 fit_parameters=["m","b"],
                 non_fit_kwargs={"x":df.x})
    f._y_obs = df.y_obs
    f._y_std = df.y_std

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
    f.fit()
    assert f._fit_has_been_run is True

    # now fit_df should have been updated
    assert np.array_equal(f.fit_df["fixed"],[False,True])
    assert np.array_equal(f.fit_df["guess"],[0,1])
    assert np.array_equal(f.fit_df["prior_mean"],[np.nan,5],equal_nan=True)
    assert np.array_equal(f.fit_df["prior_std"],[np.nan,3],equal_nan=True)
    assert np.array_equal(f.fit_df["lower_bound"],[-10,-np.inf])
    assert np.array_equal(f.fit_df["upper_bound"],[10,np.inf])

def test_MLFitter_samples(linear_fit):

    df = linear_fit["df"]
    fcn = linear_fit["fcn"]  # def simple_linear(m,b,x): return m*x + b
    coeff = linear_fit["coeff"]
    expected_result = np.array([coeff["m"],coeff["b"]])

    f = MLFitter(some_function=fcn,
                 fit_parameters=["m","b"],
                 non_fit_kwargs={"x":df.x})
    f._y_obs = df.y_obs
    f._y_std = df.y_std

    # no samples generated
    assert f.samples is None

    # run containing fit function from base class; that sets fit_has_been_run to
    # true.
    f.fit()
    assert f._fit_has_been_run is True

    # Get samples. Make sure has right shape, right means, and that they are 
    # all unique
    samples = f.samples
    assert np.array_equal(samples.shape,[f._num_samples,2])
    assert np.allclose(np.round(np.mean(samples,axis=0)),expected_result)
    assert np.unique(samples).shape[0] == f._num_samples*2
    
    # Get samples again, which should return the same object again instead of
    # regenerating
    assert samples is f.samples

    # --------------------------------------------------------------------------
    # test singular matrix exception

    # re-run fit to clear samples
    f.fit()
    assert f._fit_has_been_run is True
    assert not hasattr(f,"_samples")

    # Hack so the jacobian is now a singular matrix. This will cause the 
    # function to throw a warning return no samples
    f._fit_result.jac = np.zeros(f._fit_result.jac.shape,dtype=float)
    
    # Should not store new samples or return them
    with pytest.warns():
        new_samples = f.samples
    assert new_samples is None
    assert not hasattr(f,"_samples")

    
def test_MLFitter___repr__():

    # Stupidly simple fitting problem. find slope
    def model_to_wrap(m=1,x=np.array([1,2,3])): return m*x

    # Run _fit_has_been_run, success branch
    f = MLFitter(some_function=model_to_wrap)
    f.fit(y_obs=np.array([2,4,6]),
          y_std=[0.1,0.1,0.1])

    out = f.__repr__().split("\n")
    assert len(out) == 14

    # hack, run _fit_has_been_run, _fit_failed branch
    f._success = False

    out = f.__repr__().split("\n")
    assert len(out) == 9    

    # Run not _fit_has_been_run
    f = MLFitter(some_function=model_to_wrap)

    out = f.__repr__().split("\n")
    assert len(out) == 5

