
import pytest

from dataprob.fitters.base import Fitter
from dataprob.model_wrapper.model_wrapper import ModelWrapper
from dataprob.model_wrapper.vector_model_wrapper import VectorModelWrapper
from dataprob.fitters.base import _pretty_zeropad_str

import numpy as np
import pandas as pd
import matplotlib

import os
import pickle
import copy

def test__pretty_zeropad_str():

    x = _pretty_zeropad_str(0)
    assert x == "s{:02d}"

    x = _pretty_zeropad_str(1)
    assert x == "s{:02d}"

    x = _pretty_zeropad_str(9)
    assert x == "s{:02d}"

    x = _pretty_zeropad_str(10)
    assert x == "s{:03d}"

    x = _pretty_zeropad_str(99)
    assert x == "s{:03d}"

    x = _pretty_zeropad_str(100)
    assert x == "s{:04d}"

def test_Fitter__init__():
    """
    Test model initialization.
    """

    # Basic test of functionality
    def test_model(m,b,x): return m*x + b
    base_kwargs = {"some_function":test_model,
                   "fit_parameters":{"m":{"guess":1},
                                     "b":{"guess":0}},
                   "non_fit_kwargs":{"x":np.arange(10)},
                   "vector_first_arg":False}

    kwargs = copy.deepcopy(base_kwargs)
    f = Fitter(**kwargs)
    
    assert f.num_obs is None
    assert f.num_params == 2
    assert f.param_df.loc["m","guess"] == 1
    assert f.param_df.loc["b","guess"] == 0
    assert issubclass(type(f._model),ModelWrapper)

    assert f._fit_has_been_run is False

    # Make sure fit_parameters, non_fit_kwargs are being passed
    def test_model(m,b,x): return m*x + b
    kwargs = copy.deepcopy(base_kwargs)
    kwargs["fit_parameters"] = {"m":{"guess":10},
                                "b":{"guess":20}}
    f = Fitter(**kwargs)
    assert f.param_df.loc["m","guess"] == 10
    assert f.param_df.loc["b","guess"] == 20
    assert np.array_equal(f._model.non_fit_kwargs["x"],np.arange(10))

    # make sure vector_first_arg is being passed
    kwargs = copy.deepcopy(base_kwargs)
    kwargs["vector_first_arg"] = True
    kwargs["fit_parameters"] = ["q","r","s"]
    f = Fitter(**kwargs)
    assert issubclass(type(f._model),VectorModelWrapper)
    assert len(f.param_df) == 3
    assert np.array_equal(f.param_df["name"],["q","r","s"])
    assert len(f._model._non_fit_kwargs) == 2
    assert f._model.non_fit_kwargs["b"] is None
    assert np.array_equal(f._model.non_fit_kwargs["x"],np.arange(10))

    # Send in pre-wrapped model
    def test_model(m=10,b=1,x=[]): return m*x + b
    mw = ModelWrapper(test_model,
                      non_fit_kwargs={"x":np.arange(10)})
    f = Fitter(some_function=mw)
    assert f.param_df.loc["m","guess"] == 10
    assert f.param_df.loc["b","guess"] == 1

def test_Fitter__sanity_check():
    
    def test_model(m,b,x): return m*x + b
    base_kwargs = {"some_function":test_model,
                   "fit_parameters":{"m":{"guess":1},
                                     "b":{"guess":0}},
                   "non_fit_kwargs":{"x":np.arange(10)},
                   "vector_first_arg":False}

    kwargs = copy.deepcopy(base_kwargs)
    f = Fitter(**kwargs)

    # should always work
    f._sanity_check("some error",["fit_has_been_run"])

    # Won't work
    with pytest.raises(RuntimeError):
        f._sanity_check("some error",["not_an_attribute"])

    # None check
    f._test_attribute = None
    with pytest.raises(RuntimeError):
        f._sanity_check("some error",["test_attribute"])


def test_Fitter__process_obs_args():

    def test_model(m,b,x): return m*x + b
    base_kwargs = {"some_function":test_model,
                   "fit_parameters":{"m":{"guess":1},
                                     "b":{"guess":0}},
                   "non_fit_kwargs":{"x":np.arange(10)},
                   "vector_first_arg":False}

    kwargs = copy.deepcopy(base_kwargs)
    f_base = Fitter(**kwargs)
    
    # ----------------------------------------------------------------------
    # basic check that it runs
    f = copy.deepcopy(f_base)
    assert f.y_obs is None
    assert f.y_std is None

    f._process_obs_args(y_obs=[1,2,3],
                        y_std=[1,1,1])
    
    assert np.array_equal(f.y_obs,[1,2,3])
    assert np.array_equal(f.y_std,[1,1,1])
 
    # ----------------------------------------------------------------------
    # No y_obs, fail
    f = copy.deepcopy(f_base)
    assert f.y_obs is None
    assert f.y_std is None

    # Fail and make sure nothing changed
    with pytest.raises(ValueError):
        f._process_obs_args(y_obs=None,
                            y_std=[1,1,1])
    assert f.y_obs is None
    assert f.y_std is None

    # now set y_obs with setter
    f.y_obs = [1,2,3]
    f._process_obs_args(y_obs=None,
                        y_std=[1,1,1])
    assert np.array_equal(f.y_obs,[1,2,3])
    assert np.array_equal(f.y_std,[1,1,1])

    # ----------------------------------------------------------------------
    # No y_std, warn
    f = copy.deepcopy(f_base)
    assert f.y_obs is None
    assert f.y_std is None

    # Fail and make sure nothing changed
    with pytest.warns():
        f._process_obs_args(y_obs=[1,2,3],
                            y_std=None)
    assert np.array_equal(f.y_obs,[1,2,3])
    expected_y_std = np.mean([1,2,3])*0.1*np.ones(3)
    assert np.array_equal(f.y_std,expected_y_std)

    # ----------------------------------------------------------------------
    # y_std via setter, warn
    f = copy.deepcopy(f_base)
    assert f.y_obs is None
    assert f.y_std is None
    f.y_obs = [4,5,6] # have to define y_obs before y_std
    f.y_std = [1,1,1]

    # Should work fine, no warning because y_std defined previously
    f._process_obs_args(y_obs=[1,2,3],y_std=None)
    assert np.array_equal(f.y_obs,[1,2,3])
    assert np.array_equal(f.y_std,[1,1,1])

    # ----------------------------------------------------------------------
    # setters do sanity checking; don't test exhaustively but make sure the
    # checks are running

    # nan in y_obs
    f = copy.deepcopy(f_base)
    assert f.y_obs is None
    assert f.y_std is None
    with pytest.raises(ValueError):
        f._process_obs_args(y_obs=[np.nan,2,3],
                            y_std=[1,1,1])
        
    # negative value in y_std
    f = copy.deepcopy(f_base)
    assert f.y_obs is None
    assert f.y_std is None
    with pytest.raises(ValueError):
        f._process_obs_args(y_obs=[1,2,3],
                            y_std=[-1,1,1])

    # ----------------------------------------------------------------------
    # make sure y_std expands appropriately (done by setter, so quick check)

    f = copy.deepcopy(f_base)
    assert f.y_obs is None
    assert f.y_std is None
    f._process_obs_args(y_obs=[1,2,3],
                        y_std=1)
    assert np.array_equal(f.y_obs,[1,2,3])
    assert np.array_equal(f.y_std,[1,1,1])



def test_Fitter_fit(linear_fit):

    df = linear_fit["df"]
    fcn = linear_fit["fcn"]  # def simple_linear(m,b,x): return m*x + b

    base_kwargs = {"y_obs":df.y_obs,
                   "y_std":df.y_std,
                   "fit_kwarg":5}

    def new_fitter():

        # Create a fitter with a model, then hacked _fit, _fit_result, 
        # and _update_fit_df
        f = Fitter(some_function=fcn)
        f._fit = lambda **kwargs: None
        f._fit_result = {}
        f._update_fit_df = lambda *args: None

        return f
    
    f = new_fitter()
    kwargs = copy.deepcopy(base_kwargs)
    kwargs["names"] = None
    
    assert not hasattr(f,"_success")
    assert f._fit_has_been_run is False

    f.fit(**kwargs)

    assert f._success is None
    assert f._fit_has_been_run is True

    # ----------------------------------------------------------------------
    # make sure _process_obs_args is running with incompatible y_obs argument
    
    f = new_fitter()
    kwargs = copy.deepcopy(base_kwargs)
    kwargs["y_obs"] = [1,2,3,4]
    with pytest.raises(ValueError):
        f.fit(**kwargs)
    
    f = new_fitter() # have to reset fitter b/c model set above
    kwargs["y_obs"] = df["y_obs"]
    f.fit(**kwargs)
    
def test_Fitter__fit():

    def test_model(m,b,x): return m*x + b
    base_kwargs = {"some_function":test_model,
                   "fit_parameters":{"m":{"guess":1},
                                     "b":{"guess":0}},
                   "non_fit_kwargs":{"x":np.arange(10)},
                   "vector_first_arg":False}

    kwargs = copy.deepcopy(base_kwargs)
    f_base = Fitter(**kwargs)

    f = copy.deepcopy(f_base)
    with pytest.raises(NotImplementedError):
        f._fit()

def test_Fitter__unweighted_residuals(binding_curve_test_data):
    """
    Test unweighted residuals call against "manual" code used to generate
    test data. Just make sure answer is right; no error checking on this 
    function. 
    """

    df = binding_curve_test_data["df"]
    input_params = binding_curve_test_data["input_params"]
    f_base = Fitter(some_function=binding_curve_test_data["wrappable_model"],
                    non_fit_kwargs={"df":df})
    f = copy.deepcopy(f_base)
    
    # Calculate residual given the input params
    f.y_obs = df.Y
    r = f._unweighted_residuals(input_params)

    assert np.allclose(r,df.residual)

def test_Fitter_unweighted_residuals(binding_curve_test_data):
    """
    Test unweighted residuals call against "manual" code used to generate
    test data.
    """

    df = binding_curve_test_data["df"]
    input_params = binding_curve_test_data["input_params"]
    f_base = Fitter(some_function=binding_curve_test_data["wrappable_model"],
                    non_fit_kwargs={"df":df})
    f = copy.deepcopy(f_base)
    
    # Should fail, haven't loaded y_obs or y_std yet
    with pytest.raises(RuntimeError):
        f.unweighted_residuals(input_params)

    # Should work now
    f.y_obs = df.Y
    r = f.unweighted_residuals(input_params)
    assert np.allclose(r,df.residual)

    # Make sure error check is running
    with pytest.raises(ValueError):
        f.unweighted_residuals([1,2])

def test_Fitter__weighted_residuals(binding_curve_test_data):
    """
    Test weighted residuals call against "manual" code used to generate
    test data. Just make sure answer is right; no error checking on this 
    function. 
    """

    # Build model
    df = binding_curve_test_data["df"]
    input_params = binding_curve_test_data["input_params"]
    f_base = Fitter(some_function=binding_curve_test_data["wrappable_model"],
                    non_fit_kwargs={"df":df})
    f = copy.deepcopy(f_base)

    f.y_obs = df.Y
    f.y_std = df.Y_stdev

    # Calculate residual given the input params
    input_params = binding_curve_test_data["input_params"]
    r = f._weighted_residuals(input_params)

    assert np.allclose(r,df.weighted_residual)


def test_Fitter_weighted_residuals(binding_curve_test_data):
    """
    Test weighted residuals call against "manual" code used to generate
    test data.
    """

    df = binding_curve_test_data["df"]
    input_params = binding_curve_test_data["input_params"]
    f_base = Fitter(some_function=binding_curve_test_data["wrappable_model"],
                    non_fit_kwargs={"df":df})
    f = copy.deepcopy(f_base)

    # Should fail, haven't loaded y_obs or y_std yet
    with pytest.raises(RuntimeError):
        f.weighted_residuals(input_params)

    f.y_obs = df.Y

    # Should fail, haven't loaded y_std yet
    with pytest.raises(RuntimeError):
        f.weighted_residuals(input_params)

    f.y_std = df.Y_stdev

    r = f.weighted_residuals(input_params)
    assert np.allclose(r,df.weighted_residual)

    # Make sure error check is running
    with pytest.raises(ValueError):
        f.weighted_residuals([1,2])

def test_Fitter__ln_like(binding_curve_test_data):
    """
    Test internal function -- no error checking. 
    """

    df = binding_curve_test_data["df"]
    input_params = binding_curve_test_data["input_params"]
    f_base = Fitter(some_function=binding_curve_test_data["wrappable_model"],
                    non_fit_kwargs={"df":df})
    f = copy.deepcopy(f_base)

    f.y_obs = df.Y
    f.y_std = df.Y_stdev

    input_params = binding_curve_test_data["input_params"]

    L = f.ln_like(input_params)
    assert np.allclose(L,binding_curve_test_data["ln_like"])

def test_Fitter_ln_like(binding_curve_test_data):
    """
    Test log likelihood call against "manual" code used to generate
    test data.
    """

    df = binding_curve_test_data["df"]
    input_params = binding_curve_test_data["input_params"]
    f_base = Fitter(some_function=binding_curve_test_data["wrappable_model"],
                    non_fit_kwargs={"df":df})
    f = copy.deepcopy(f_base)

    # Should fail, haven't loaded y_obs or y_std yet
    with pytest.raises(RuntimeError):
        f.ln_like(input_params)

    f.y_obs = df.Y

    # Should fail, haven't loaded y_std yet
    with pytest.raises(RuntimeError):
        f.ln_like(input_params)

    f.y_std = df.Y_stdev
    
    L = f.ln_like(input_params)
    
    assert np.allclose(L,binding_curve_test_data["ln_like"])

    # make sure input params sanity check is running
    with pytest.raises(ValueError):
        f.ln_like([1,2])


# ---------------------------------------------------------------------------- #
# Test setters, getters, and internal sanity checks
# ---------------------------------------------------------------------------- #

def test_Fitter_y_obs_setter_getter(binding_curve_test_data):
    """
    Test the y_obs setter.
    """

    def test_fcn(x=1,y=2): return x*y

    f = Fitter(some_function=test_fcn)
 
    y_obs_input = np.array(binding_curve_test_data["df"].Y)

    f.y_obs = y_obs_input
    assert f.y_obs is not None
    assert np.array_equal(f.y_obs,y_obs_input)
    assert f._fit_has_been_run is False

    f = Fitter(some_function=test_fcn)
    with pytest.raises(ValueError):
        f.y_obs = "a"
    with pytest.raises(ValueError):
        f.y_obs = ["a","b"]

    # nan
    tmp_input = y_obs_input.copy()
    tmp_input[0] = np.nan
    with pytest.raises(ValueError):
        f.y_std = tmp_input

    f = Fitter(some_function=test_fcn)
    f.y_obs = y_obs_input
    assert np.array_equal(f.y_obs,y_obs_input)
    assert f.num_obs == y_obs_input.shape[0]

  
def test_Fitter_y_std_setter_getter(binding_curve_test_data):
    """
    Test the y_std setter.
    """

    def test_fcn(x=1,y=2): return x*y
    y_obs_input = np.array(binding_curve_test_data["df"].Y)
    y_std_input = np.array(binding_curve_test_data["df"].Y_stdev)

    f = Fitter(some_function=test_fcn)
    assert f.y_std is None
    assert f.num_obs is None

    # Cannot add y_std before y_obs
    with pytest.raises(ValueError):
        f.y_std = y_std_input

    f.y_obs = y_obs_input

    f.y_std = y_std_input
    assert np.array_equal(y_std_input,f.y_std)
    assert f.num_obs == len(y_std_input)
    assert f._fit_has_been_run is False

    f = Fitter(some_function=test_fcn)
    assert f.y_std is None
    f.y_obs = y_obs_input

    # Bad values
    with pytest.raises(ValueError):
        f.y_std = "a"
    with pytest.raises(ValueError):
        f.y_std = ["a","b"]
    with pytest.raises(ValueError):
        f.y_std = y_std_input[:-1]
    
    # nan
    tmp_input = y_std_input.copy()
    tmp_input[0] = np.nan
    with pytest.raises(ValueError):
        f.y_std = tmp_input
    
    # negative
    tmp_input = y_std_input.copy()
    tmp_input[0] = -1
    with pytest.raises(ValueError):
        f.y_std = tmp_input
    
    f.y_std = y_std_input
    assert np.array_equal(y_std_input,f.y_std)
    assert f._fit_has_been_run is False

    # Set single value
    f = Fitter(some_function=test_fcn)
    assert f.y_std is None
    f.y_obs = y_obs_input
    f.y_std = 1.0
    assert np.array_equal(f.y_std,np.ones(f.y_obs.shape))


def test_Fitter_param_df():
    
    def test_fcn(a=1,b=2): return a*b
    f = Fitter(some_function=test_fcn)
    
    assert len(f.param_df) == 2
    assert np.array_equal(f.param_df["name"],["a","b"])


def test_Fitter_non_fit_kwargs():

    def test_fcn(a=1,b=2,c="test"): return a*b
    f = Fitter(some_function=test_fcn)
    assert len(f.non_fit_kwargs) == 1
    assert f.non_fit_kwargs["c"] == "test"
    f.non_fit_kwargs["c"] = "something_else"
    assert f.non_fit_kwargs["c"] == "something_else"
    
    # should work
    f._model.finalize_params()
    
    # should fail
    f.non_fit_kwargs.pop("c")
    with pytest.raises(ValueError):
        f._model.finalize_params()

    # should work
    f.non_fit_kwargs["c"] = 14
    f._model.finalize_params()


def test_Fitter_data_df():
    
    def test_fcn(a=1,b=2): return a*b
    f = Fitter(some_function=test_fcn)

    out_df = f.data_df
    assert len(out_df) == 0
    
    y_obs = np.arange(10,dtype=float)
    y_std = np.ones(10,dtype=float)
    y_calc = np.arange(10)*0.9

    f = Fitter(some_function=test_fcn)
    f.y_obs = y_obs
    out_df = f.data_df
    assert len(out_df) == 10
    assert len(out_df.columns) == 1
    assert np.array_equal(out_df["y_obs"],y_obs)

    f = Fitter(some_function=test_fcn)
    f.y_obs = y_obs
    f.y_std = y_std
    out_df = f.data_df
    assert len(out_df) == 10
    assert len(out_df.columns) == 2
    assert np.array_equal(out_df["y_obs"],y_obs)
    assert np.array_equal(out_df["y_std"],y_std)
    
    def hack_fcn(a,b): return np.arange(10)*0.9
    f = Fitter(some_function=hack_fcn)
    f.y_obs = y_obs
    f.y_std = y_std

    # hack it so it thinks its done
    f._success = True
    f._fit_df = {"estimate":[1,2]}
    f.model([2,3])

    # check final data_df
    out_df = f.data_df
    assert len(out_df) == 10
    assert len(out_df.columns) == 5
    assert np.array_equal(out_df["y_obs"],y_obs)
    assert np.array_equal(out_df["y_std"],y_std)
    assert np.array_equal(out_df["y_calc"],y_calc)
    assert np.array_equal(out_df["unweighted_residuals"],
                          y_calc - y_obs)
    assert np.array_equal(out_df["weighted_residuals"],
                          (y_calc - y_obs)/y_std)


def test_Fitter__initialize_fit_df():
    
    # test on fake class
    class TestClass:
        def __init__(self):
            self.param_df = {"name":["a","b"],
                             "guess":[10,10],
                             "fixed":[True,False],
                             "lower_bound":[-np.inf,0],
                             "upper_bound":[np.inf,100],
                             "prior_mean":[1,np.nan],
                             "prior_std":[1,np.nan]}
    
    tc = TestClass()
    Fitter._initialize_fit_df(tc)
    assert np.array_equal(tc.param_df["name"],tc._fit_df["name"])
    assert np.sum(np.isnan(tc._fit_df["estimate"]))
    assert np.sum(np.isnan(tc._fit_df["std"]))
    assert np.sum(np.isnan(tc._fit_df["low_95"]))
    assert np.sum(np.isnan(tc._fit_df["high_95"]))
    
    columns = ["guess","fixed",
               "lower_bound","upper_bound",
               "prior_mean","prior_std"]
    for k in columns:
        assert np.array_equal(tc.param_df[k],tc._fit_df[k],equal_nan=True)
    

def test_Fitter__update_fit_df():

    def test_fcn(a=1,b=2): return a*b
    f = Fitter(some_function=test_fcn)
    with pytest.raises(NotImplementedError):
        f._update_fit_df()

def test_Fitter_fit_df():

    # This checks initialization. Need to write implementation-specific tests

    def test_fcn(a=1,b=2,x="array"): return x*a + b
    x = np.arange(10)
    y_obs = x*2 + 1
    
    f = Fitter(some_function=test_fcn,
               non_fit_kwargs={"x":x})
    
    assert len(f.fit_df) == 2
    assert np.array_equal(f.fit_df["name"],["a","b"])
    assert np.array_equal(f.fit_df.columns,
                          ["name","estimate","std","low_95","high_95",
                           "guess","fixed","lower_bound","upper_bound",
                           "prior_mean","prior_std"])

    f.y_obs = y_obs
    f.y_std = 0.1


def test_Fitter_samples():
    
    def test_fcn(a=1,b=2,x="array"): return x*a + b
    f = Fitter(some_function=test_fcn)
    assert f.samples is None
    f._samples = "something"
    assert f.samples == "something"


def test_Fitter_get_sample_df():
    
    # some test data
    y_obs = np.arange(10)
    y_std = np.ones(10)
    def test_fcn(a=1,b=2): return a*b*np.ones(10)
    fake_samples = np.ones((1000,2),dtype=float)

    # Error checking on num_samples
    f = Fitter(some_function=test_fcn)
    with pytest.raises(ValueError):
        f.get_sample_df(num_samples=-1)
    with pytest.raises(ValueError):
        f.get_sample_df(num_samples="a")
        
    # empty class - return empty dataframe
    f = Fitter(some_function=test_fcn)
    sample_df = f.get_sample_df()
    assert issubclass(type(sample_df),pd.DataFrame)
    assert len(sample_df) == 0

    # add y_obs, should be in dataframe by itself
    f.y_obs = y_obs
    sample_df = f.get_sample_df()
    assert issubclass(type(sample_df),pd.DataFrame)
    assert len(sample_df) == 10
    assert np.array_equal(sample_df["y_obs"],y_obs)
    assert np.array_equal(sample_df.columns,["y_obs"])

    # add y_std, should now be in dataframe
    f.y_std = y_std
    sample_df = f.get_sample_df()
    assert issubclass(type(sample_df),pd.DataFrame)
    assert len(sample_df) == 10
    assert np.array_equal(sample_df["y_obs"],y_obs)
    assert np.array_equal(sample_df["y_std"],y_std)
    assert np.array_equal(sample_df.columns,["y_obs","y_std"])

    # Create a fitter that has apparently been run, but has no samples
    f = Fitter(some_function=test_fcn)
    f.y_obs = y_obs
    f.y_std = y_std
    f._fit_df = pd.DataFrame({"estimate":[10,20]})
    f._success = True

    sample_df = f.get_sample_df()
    assert issubclass(type(sample_df),pd.DataFrame)
    assert len(sample_df) == 10
    assert np.array_equal(sample_df["y_obs"],y_obs)
    assert np.array_equal(sample_df["y_std"],y_std)
    expected_y_calc = 10*20*np.ones(10)
    assert np.array_equal(sample_df["y_calc"],expected_y_calc)
    assert np.array_equal(sample_df.columns,["y_obs","y_std","y_calc"])

    # Add some fake samples
    f._samples = fake_samples
    sample_df = f.get_sample_df()
    assert len(sample_df) == 10
    assert np.array_equal(sample_df["y_obs"],y_obs)
    assert np.array_equal(sample_df["y_std"],y_std)
    expected_y_calc = 10*20*np.ones(10)
    assert np.array_equal(sample_df["y_calc"],expected_y_calc)
    assert np.array_equal(sample_df.columns[:4],["y_obs","y_std","y_calc","s00000"])
    assert sample_df.columns[-1] == "s00990"
    assert len(sample_df.columns) == 103

    # Get fewer samples
    sample_df = f.get_sample_df(num_samples=10)
    assert len(sample_df) == 10
    assert np.array_equal(sample_df["y_obs"],y_obs)
    assert np.array_equal(sample_df["y_std"],y_std)
    expected_y_calc = 10*20*np.ones(10)
    assert np.array_equal(sample_df["y_calc"],expected_y_calc)
    assert np.array_equal(sample_df.columns[:4],["y_obs","y_std","y_calc","s00000"])
    assert sample_df.columns[-1] == "s00999"
    assert len(sample_df.columns) == 13


def test_Fitter_write_samples(tmpdir):
    
    cwd = os.getcwd()
    os.chdir(tmpdir)

    test_file = "test-out.pickle"

    def test_fcn(a,b,c,d,e): return a*b

    # Should not write out because samples do not exist yet
    f = Fitter(some_function=test_fcn)
    assert f.samples is None
    assert not os.path.exists(test_file)
    f.write_samples(test_file)
    assert not os.path.exists(test_file)

    # create fake samples and write out
    f._samples = np.ones((100,5),dtype=float)
    assert not f.samples is None
    assert not os.path.exists(test_file)
    f.write_samples(test_file)
    assert os.path.exists(test_file)

    # read samples back in to make sure they wrote as a pickle
    with open(test_file,"rb") as handle:
        read_back = pickle.load(handle)
    assert np.array_equal(read_back,f._samples)

    # Try and fail to write samples to an existing pickle files
    with open("existing-file.pickle","w") as g:
        g.write("yo")
    
    with pytest.raises(FileExistsError):
        f.write_samples("existing-file.pickle")

    os.chdir(cwd)

def test_Fitter_append_samples(tmpdir):

    cwd = os.getcwd()
    os.chdir(tmpdir)

    # -----------------------------------------------------------------------
    # make some files and arrays for testing

    sample_array = np.ones((100,3),dtype=float)
    with open("test.pickle","wb") as p:
        pickle.dump(sample_array,p)
    with open("bad_file.txt","w") as g:
        g.write("yo")

    # -----------------------------------------------------------------------
    # Build a hacked Fitter object that has existing samples, three params, 
    # and an overwritten _update_fit_df call that does nothing.

    # Create a three parameter model to assign to the fitter (setting the 
    # number of parameters)
    def test_fcn(a,b,c): return a*b*c

    # Create fitter and assign model
    base_f = Fitter(some_function=test_fcn)

    # Assign samples
    base_f._samples = sample_array.copy()
    assert np.array_equal(base_f.samples.shape,(100,3))

    # add dummy function
    def dummy(*args,**kwargs): pass
    base_f._update_fit_df = dummy

    # -----------------------------------------------------------------------
    # Run tests

    # Nothing happens
    f = Fitter(some_function=test_fcn)
    f.append_samples(sample_file=None,
                     sample_array=None)
    
    # Check for existing samples (should fail without samples)
    f = Fitter(some_function=test_fcn)
    assert f.samples is None
    with pytest.raises(ValueError):
        f.append_samples(sample_array=sample_array)
    f = copy.deepcopy(base_f)
    f.append_samples(sample_array=sample_array)

    # Too many inputs
    with pytest.raises(ValueError):
        f.append_samples(sample_file="test.pickle",
                         sample_array=sample_array)

    f = copy.deepcopy(base_f)
    assert np.array_equal(f.samples.shape,(100,3))
    
    f.append_samples(sample_file="test.pickle")
    assert np.array_equal(f.samples.shape,(200,3))

    f.append_samples(sample_file="test.pickle")
    assert np.array_equal(f.samples.shape,(300,3))

    f.append_samples(sample_array=sample_array)
    assert np.array_equal(f.samples.shape,(400,3))

    # Bad files
    f = copy.deepcopy(base_f)
    with pytest.raises(FileNotFoundError):
        f.append_samples(sample_file="not_real_file")
    with pytest.raises(pickle.UnpicklingError):
        f.append_samples(sample_file="bad_file.txt")

    # not coercable to floats
    f = copy.deepcopy(base_f)
    not_coercable_to_float = [["a","b","c"],
                              ["a","b","c"],
                              ["a","b","c"]]
    with pytest.raises(ValueError):
        f.append_samples(sample_array=not_coercable_to_float)

    # vector with wrong number of dimensions
    f = copy.deepcopy(base_f)
    with pytest.raises(ValueError):
        f.append_samples(sample_array=np.ones(3,dtype=float))
    
    # array with wrong number of parameter dimensions
    f = copy.deepcopy(base_f)
    with pytest.raises(ValueError):
        f.append_samples(sample_array=np.ones((100,2),dtype=float))

    # Make sure update estimates is running by hacking f._update_estimates to
    # throw an exception
    f = copy.deepcopy(base_f)
    assert np.array_equal(f.samples.shape,(100,3))
    f.append_samples(sample_array=sample_array)
    assert np.array_equal(f.samples.shape,(200,3))
    
    def dummy(*args,**kwargs):
        raise RuntimeError
    f._update_fit_df = dummy
    
    with pytest.raises(RuntimeError):
        f.append_samples(sample_array=sample_array)

    # -----------------------------------------------------------------------
    # Run test with a fixed parameter

    def test_fcn(a,b,c): return a*b*c
    def dummy(*args,**kwargs): pass
    f = Fitter(some_function=test_fcn)
    f.param_df.loc["a","fixed"] = True
    
    # some hacks to put this into a state to append samples
    f._samples = np.ones((100,2),dtype=float)
    f._update_fit_df = dummy

    with pytest.raises(ValueError):
        f.append_samples(sample_array=np.ones((100,3),dtype=float))
    
    f.append_samples(sample_array=np.ones((100,2),dtype=float))

    os.chdir(cwd)

def test_Fitter_num_params(binding_curve_test_data):

    def test_fcn(a=2,b=3): return a*b
    f = Fitter(some_function=test_fcn)
    assert f.num_params == 2

    assert f.model() == 2*3

    with pytest.raises(ValueError):
        f.model([7,8,9])

    def test_fcn(a=2,b=3,c=4): return a*b*c
    f = Fitter(some_function=test_fcn)
    assert f.num_params == 3

    assert f.model() == 2*3*4

    with pytest.raises(ValueError):
        f.model([7,8,9,10])

    def test_fcn(a=2,b=3,c=4): return a*b*c
    f = Fitter(some_function=test_fcn)
    assert f.num_params == 3
    
    f.param_df.loc["a","fixed"] = True
    f._model.finalize_params()
    assert f.num_params == 2

    f.param_df.loc["b","fixed"] = True
    f._model.finalize_params()
    assert f.num_params == 1


def test_Fitter_num_obs():

    def test_fcn(a=2,b=3): return a*b
    f = Fitter(some_function=test_fcn)
    assert f.num_obs is None

    f.y_obs = np.arange(10)
    assert f.num_obs == 10

    f = Fitter(some_function=test_fcn)
    f.y_obs = np.array([])
    assert f.num_obs == 0

def test_Fitter_success():
    
    def test_fcn(a=2,b=3,c=4): return a*b*c
    f = Fitter(some_function=test_fcn)
    assert f.success is None
    f._success = True
    assert f.success is True

def test_Fitter_fit_info():

    def test_fcn(a=2,b=3,c=4): return a*b*c
    f = Fitter(some_function=test_fcn)
    with pytest.raises(NotImplementedError):
        f.fit_info

def test_Fitter_fit_result():
    
    def test_fcn(a=2,b=3,c=4): return a*b*c
    f = Fitter(some_function=test_fcn)
    assert f.fit_result is None
    f._fit_result = "something"
    assert f.fit_result == "something"
