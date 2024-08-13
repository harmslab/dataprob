import pytest

from dataprob.fitters.ml import MLFitter
from dataprob.model_wrapper.model_wrapper import ModelWrapper

import numpy as np
import pandas as pd

def test_MLFitter___init__():

    f = MLFitter(num_samples=100)
    assert f.fit_type == "maximum likelihood"
    assert f._num_samples == 100

def test_MLFitter_fit(linear_fit):

    # Basic functionality and logic tests. Numerical tests on more interesting
    # fitting problems are below. Tests run through .fit() because that
    # initializes everything then calls ._fit(). See the base-class for that. 

    # --------------------------------------------------------------------------
    # Simple model to test

    df = linear_fit["df"]
    fcn = linear_fit["fcn"]  # def simple_linear(m,b,x): return m*x + b
    linear_mw = ModelWrapper(fcn,fittable_params=["m","b"])
    linear_mw.x = df.x
    coeff = linear_fit["coeff"]
    expected_result = np.array([coeff["m"],coeff["b"]])

    # --------------------------------------------------------------------------
    # Run fit

    f = MLFitter()
    f.model = linear_mw
    f.y_obs = df.y_obs
    f.y_std = df.y_std
    
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


def test_MLFitter__update_fit_df(linear_fit):
    
    df = linear_fit["df"]
    fcn = linear_fit["fcn"]  # def simple_linear(m,b,x): return m*x + b
    linear_mw = ModelWrapper(fcn,fittable_params=["m","b"])
    linear_mw.x = df.x
    coeff = linear_fit["coeff"]
    expected_result = np.array([coeff["m"],coeff["b"]])

    f = MLFitter()
    f.model = linear_mw
    f.y_obs = df.y_obs
    f.y_std = df.y_std

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
    

def test_MLFitter_samples(linear_fit):

    df = linear_fit["df"]
    fcn = linear_fit["fcn"]  # def simple_linear(m,b,x): return m*x + b
    linear_mw = ModelWrapper(fcn,fittable_params=["m","b"])
    linear_mw.x = df.x
    coeff = linear_fit["coeff"]
    expected_result = np.array([coeff["m"],coeff["b"]])

    f = MLFitter()
    f.model = linear_mw
    f.y_obs = df.y_obs
    f.y_std = df.y_std

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
    mw = ModelWrapper(model_to_fit=model_to_wrap)

    # Run _fit_has_been_run, success branch
    f = MLFitter()
    f.model = mw
    f.fit(y_obs=np.array([2,4,6]),
          y_std=[0.1,0.1,0.1])

    out = f.__repr__().split("\n")
    assert len(out) == 14

    # hack, run _fit_has_been_run, _fit_failed branch
    f._success = False

    out = f.__repr__().split("\n")
    assert len(out) == 9    

    # Run not _fit_has_been_run
    f = MLFitter()
    f.model = mw

    out = f.__repr__().split("\n")
    assert len(out) == 5

    



def xtest_MLFitter_fit(binding_curve_test_data,fit_tolerance_fixture):
    """
    Test the ability to fit the test data in binding_curve_test_data.
    """


    pass

    # Do fit using a generic model and then creating and using a ModelWrapper
    # around wrappable_model


    return

    for model_key in ["generic_model","wrappable_model"]:

        f = MLFitter()
        model = binding_curve_test_data[model_key]
        guesses = binding_curve_test_data["guesses"]
        df = binding_curve_test_data["df"]
        input_params = np.array(binding_curve_test_data["input_params"])

        if model_key == "wrappable_model":
            mw = ModelWrapper(model)
            mw.df = df
            mw.param_df["K","lower_bound"] = 0
            mw.param_df["K","upper_bound"] = 10
    
        f.fit(model=model,guesses=guesses,y_obs=df.Y,y_std=df.Y_stdev)

        # Assert that we succesfully passed in bounds
        assert np.allclose(f.param_df["lower_bound"],[0])
        assert np.allclose(f.param_df["upper_bound"],[0])
        
        # Make sure fit worked
        assert f.success

        # Make sure fit gave right answer
        assert np.allclose(f.fit_df["estimate"],
                           input_params,
                           rtol=fit_tolerance_fixture,
                           atol=fit_tolerance_fixture*input_params)

        # Make sure mean of sampled uncertainty gives right answer
        sampled = np.mean(f.samples,axis=0)
        assert np.allclose(f.fit_df["estimate"],
                           sampled,
                           rtol=fit_tolerance_fixture,
                           atol=fit_tolerance_fixture*input_params)

        # Make sure corner plot call works and generates a plot
        corner_fig = f.corner_plot()
        assert corner_fig is not None

        # Make sure data frame that comes out is correct
        df = f.fit_df

        assert isinstance(df,pd.DataFrame)
        assert np.allclose(df["estimate"].iloc[:],
                           input_params,
                           rtol=fit_tolerance_fixture,
                           atol=fit_tolerance_fixture*input_params)
        assert np.array_equal(df["name"],f.fit_df["name"])
        assert np.array_equal(df["estimate"],f.fit_df["estimate"])
        assert np.array_equal(df["std"],f.fit_df["std"])
        assert np.array_equal(df["low_95"],f.fit_df["low_95"])
        assert np.array_equal(df["high_95"],f.fit_df["high_95"])
        # assert np.array_equal(df["guess"],f.param_df["guess"])
        # assert np.array_equal(df["lower_bound"],f.param_df["lower_bound"])
        # assert np.array_equal(df["upper_bound"],f.param_df["upper_bound"])





