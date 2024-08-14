
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

    f = Fitter()
    
    assert f.num_obs is None
    assert f.num_params is None

    assert f._fit_has_been_run is False
    assert f._fit_type == ""

def test_Fitter__sanity_check():
    
    f = Fitter()
    f._sanity_check("some error",["fit_type"])
    with pytest.raises(RuntimeError):
        f._sanity_check("some error",["not_an_attribute"])

    # None check
    f._test_attribute = None
    with pytest.raises(RuntimeError):
        f._sanity_check("some error",["test_attribute"])

def test_Fitter__process_model_args():

    # Create a fitter that already has a model
    f = Fitter()
    def test_fcn(a=1,b=2): return a*b
    mw = ModelWrapper(model_to_fit=test_fcn)
    f.model = mw

    # Should run. 
    f._process_model_args(model=None,guesses=None,names=None)
    assert f._model is mw

    # Die. Cannot specify a new model or names
    with pytest.raises(ValueError):
        f._process_model_args(model=test_fcn,guesses=[1,2],names=["a","b"])

    # Die. Cannot specify a new model
    with pytest.raises(ValueError):
        f._process_model_args(model=test_fcn,guesses=None,names=None)

    # Die. Cannot specify a new names
    with pytest.raises(ValueError):
        f._process_model_args(model=None,guesses=None,names=["a","b"])

    # Create an empty fitter
    f = Fitter()
    with pytest.raises(AttributeError):
        f._model

    # No model sent in, die.
    with pytest.raises(ValueError):
        f._process_model_args(model=None,guesses=[1,2],names=["a","b"])

    # Extra names sent in. Die
    with pytest.raises(ValueError):
        f._process_model_args(model=mw,guesses=[1,2],names=["a","b"])
    
    # model and guesses -- fine
    f._process_model_args(model=mw,guesses=[1,2],names=None)    
    assert f._model is mw

    # Model and no guesses, fine. 
    f = Fitter()
    with pytest.raises(AttributeError):
        f._model
    f._process_model_args(model=mw,guesses=None,names=None)    
    assert f._model is mw

    # Send in naked function, a is a length-two list
    def test_fcn(a,b=2): return a[0]*a[1]*b
    f = Fitter()
    with pytest.raises(AttributeError):
        f._model
    
    # Naked function. Die because no guesses or names specified. 
    with pytest.raises(ValueError):
        f._process_model_args(model=test_fcn,guesses=None,names=None)
    
    # Naked function. Die because no guesses or names specified. 
    with pytest.raises(ValueError):
        f._process_model_args(model=test_fcn,guesses=None,names=None)
    
    # Naked function. Die because no guesses  specified. 
    with pytest.raises(ValueError):
        f._process_model_args(model=test_fcn,guesses=None,names=["x","y"])

    # Naked function. Work. 
    f._process_model_args(model=test_fcn,guesses=[5,6],names=["x","y"])
    np.array_equal(f._model.param_df["name"],["x","y"])

    # Naked function. Work and create default parameter names
    f = Fitter()
    f._process_model_args(model=test_fcn,guesses=[5,6],names=None)
    np.array_equal(f._model.param_df["name"],["p0","p1"])

    # Naked function. die because guesses and names have different lengths
    f = Fitter()
    with pytest.raises(ValueError):
        f._process_model_args(model=test_fcn,guesses=[5,6],names=["x","y","z"])

def test_Fitter__process_fit_args():

    f_base = Fitter()
    def test_fcn(a=5,b=6): return a*b
    mw = ModelWrapper(test_fcn)
    f_base.model = mw
    
    base_kwargs = {"guesses":[1,2],
                   "lower_bounds":[-10,-20],
                   "upper_bounds":[10,20],
                   "prior_means":[1,np.nan],
                   "prior_stds":[1,np.nan],
                   "fixed":[False,False]}

    # basic check that it runs
    f = copy.deepcopy(f_base)
    kwargs = copy.deepcopy(base_kwargs)
    assert np.array_equal(f.param_df["guess"],[5,6])
    f._process_fit_args(**kwargs)
    assert np.array_equal(f.param_df["guess"],[1,2])
    
    # --------------------------------------------------------------------
    # guesses

    f = copy.deepcopy(f_base)
    kwargs = copy.deepcopy(base_kwargs)

    # no argument -- do nothing
    kwargs["guesses"] = None
    assert np.array_equal(f.param_df["guess"],[5,6])
    f._process_fit_args(**kwargs)
    assert np.array_equal(f.param_df["guess"],[5,6])

    # good argument
    kwargs["guesses"] = [1,2]
    assert np.array_equal(f.param_df["guess"],[5,6])
    f._process_fit_args(**kwargs)
    assert np.array_equal(f.param_df["guess"],[1,2])
    
    # too long
    kwargs["guesses"] = [1,2,3]
    with pytest.raises(ValueError):
        f._process_fit_args(**kwargs)
    
    # --------------------------------------------------------------------
    # lower_bounds

    f = copy.deepcopy(f_base)
    kwargs = copy.deepcopy(base_kwargs)

    # no argument -- do nothing
    kwargs["lower_bounds"] = None
    assert np.array_equal(f.param_df["lower_bound"],[-np.inf,-np.inf])
    f._process_fit_args(**kwargs)
    assert np.array_equal(f.param_df["lower_bound"],[-np.inf,-np.inf])

    # good argument
    kwargs["lower_bounds"] = [-10,-20]
    assert np.array_equal(f.param_df["lower_bound"],[-np.inf,-np.inf])
    f._process_fit_args(**kwargs)
    assert np.array_equal(f.param_df["lower_bound"],[-10,-20])

    # too long
    kwargs["lower_bounds"] = [-10,-20,-30]
    with pytest.raises(ValueError):
        f._process_fit_args(**kwargs)

    # --------------------------------------------------------------------
    # upper_bounds

    f = copy.deepcopy(f_base)
    kwargs = copy.deepcopy(base_kwargs)

    # no argument -- do nothing
    kwargs["upper_bounds"] = None
    assert np.array_equal(f.param_df["upper_bound"],[np.inf,np.inf])
    f._process_fit_args(**kwargs)
    assert np.array_equal(f.param_df["upper_bound"],[np.inf,np.inf])

    # good argument
    kwargs["upper_bounds"] = [10,20]
    assert np.array_equal(f.param_df["upper_bound"],[np.inf,np.inf])
    f._process_fit_args(**kwargs)
    assert np.array_equal(f.param_df["upper_bound"],[10,20])

    # too long
    kwargs["upper_bounds"] = [10,20,30]
    with pytest.raises(ValueError):
        f._process_fit_args(**kwargs)

    # --------------------------------------------------------------------
    # prior_means and prior_stds (both must be set together)

    f = copy.deepcopy(f_base)
    kwargs = copy.deepcopy(base_kwargs)

    # no arguments -- do nothing
    kwargs["prior_means"] = None
    kwargs["prior_stds"] = None
    assert np.array_equal(f.param_df["prior_mean"],[np.nan,np.nan],equal_nan=True)
    assert np.array_equal(f.param_df["prior_std"],[np.nan,np.nan],equal_nan=True)
    f._process_fit_args(**kwargs)
    assert np.array_equal(f.param_df["prior_mean"],[np.nan,np.nan],equal_nan=True)
    assert np.array_equal(f.param_df["prior_std"],[np.nan,np.nan],equal_nan=True)

    # good arguments
    kwargs["prior_means"] = [1,np.nan]
    kwargs["prior_stds"] = [2,np.nan]
    assert np.array_equal(f.param_df["prior_mean"],[np.nan,np.nan],equal_nan=True)
    assert np.array_equal(f.param_df["prior_std"],[np.nan,np.nan],equal_nan=True)

    f._process_fit_args(**kwargs)
    assert np.array_equal(f.param_df["prior_mean"],[1,np.nan],equal_nan=True)
    assert np.array_equal(f.param_df["prior_std"],[2,np.nan],equal_nan=True)

    # This won't work unless we also set prior_stds
    kwargs["prior_means"] = [np.nan,np.nan]
    with pytest.raises(ValueError):
        f._process_fit_args(**kwargs)
    kwargs["prior_stds"] = [np.nan,np.nan]
    f._process_fit_args(**kwargs)
    assert np.array_equal(f.param_df["prior_mean"],[np.nan,np.nan],equal_nan=True)
    assert np.array_equal(f.param_df["prior_std"],[np.nan,np.nan],equal_nan=True)

    # bad arguments (too long). Make sure checks are happening on both
    kwargs["prior_means"] = [1,1,1]
    kwargs["prior_stds"] = [2,2]
    with pytest.raises(ValueError):
        f._process_fit_args(**kwargs)

    kwargs["prior_means"] = [1,1]
    kwargs["prior_stds"] = [2,2,2]
    with pytest.raises(ValueError):
        f._process_fit_args(**kwargs)

    kwargs["prior_means"] = [1,1,1]
    kwargs["prior_stds"] = [2,2,2]
    with pytest.raises(ValueError):
        f._process_fit_args(**kwargs)

    # --------------------------------------------------------------------
    # fixed

    f = copy.deepcopy(f_base)
    kwargs = copy.deepcopy(base_kwargs)
    kwargs["fixed"] = None
    assert np.array_equal(f.param_df["fixed"],[False,False])
    f._process_fit_args(**kwargs)
    assert np.array_equal(f.param_df["fixed"],[False,False])

    kwargs["fixed"] = [True,False]
    assert np.array_equal(f.param_df["fixed"],[False,False])
    f._process_fit_args(**kwargs)
    assert np.array_equal(f.param_df["fixed"],[True,False])

    # too long
    kwargs["fixed"] = [True,True,True]
    with pytest.raises(ValueError):
        f._process_fit_args(**kwargs)

    # --------------------------------------------------------------------
    # verify param_df sanity checking is occuring by sending in some 
    # incompatible bounds

    f = copy.deepcopy(f_base)
    print(f.param_df)
    kwargs = copy.deepcopy(base_kwargs)
    
    assert np.array_equal(f.param_df["guess"],[5,6])
    assert np.array_equal(f.param_df["lower_bound"],[-np.inf,-np.inf])

    # Send in incompatible values and make sure it does not set
    kwargs["guesses"] = [0,0]
    kwargs["lower_bounds"] = [5,5]
    with pytest.raises(ValueError):
        f._process_fit_args(**kwargs)
    assert np.array_equal(f.param_df["guess"],[5,6])
    assert np.array_equal(f.param_df["lower_bound"],[-np.inf,-np.inf])

    # Relieve incompatibility
    kwargs["guesses"] = [6,7]
    kwargs["lower_bounds"] = [5,5]
    f._process_fit_args(**kwargs)
    assert np.array_equal(f.param_df["guess"],[6,7])
    assert np.array_equal(f.param_df["lower_bound"],[5,5])

def test_Fitter__process_obs_args():

    f_base = Fitter()
    def test_fcn(a=5,b=6): return a*b
    mw = ModelWrapper(test_fcn)
    f_base.model = mw
    
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
    linear_mw = ModelWrapper(fcn,fittable_params=["m","b"])

    base_kwargs = {"model":linear_mw,
                   "guesses":[1,2],
                   "names":["m","b"],
                   "lower_bounds":[-10,-20],
                   "upper_bounds":[10,20],
                   "prior_means":[1,np.nan],
                   "prior_stds":[1,np.nan],
                   "fixed":[False,False],
                   "y_obs":df.y_obs,
                   "y_std":df.y_std,
                   "fit_kwarg":5}

    def new_fitter():

        # Create a fitter with a model, then hacked _fit, _fit_result, 
        # and _update_fit_df
        f = Fitter()
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
    # make sure _process_model_args is running with incompatible name argument

    f = new_fitter()
    kwargs = copy.deepcopy(base_kwargs)
    with pytest.raises(ValueError):
        f.fit(**kwargs)
    kwargs["names"] = None
    f.fit(**kwargs)
    
    # ----------------------------------------------------------------------
    # make sure _process_fit_args is running with incompatible guesses argument
    
    f = new_fitter()
    kwargs = copy.deepcopy(base_kwargs)
    kwargs["names"] = None
    kwargs["guesses"] = [1,2,3]
    with pytest.raises(ValueError):
        f.fit(**kwargs)
    
    f = new_fitter() # have to reset fitter b/c model set above
    kwargs["guesses"] = [5,6]
    f.fit(**kwargs)

    # ----------------------------------------------------------------------
    # make sure _process_obs_args is running with incompatible y_obs argument
    
    f = new_fitter()
    kwargs = copy.deepcopy(base_kwargs)
    kwargs["names"] = None
    kwargs["y_obs"] = [1,2,3,4]
    with pytest.raises(ValueError):
        f.fit(**kwargs)
    
    f = new_fitter() # have to reset fitter b/c model set above
    kwargs["y_obs"] = df["y_obs"]
    f.fit(**kwargs)
    
def test_Fitter__fit():
    f = Fitter()
    with pytest.raises(NotImplementedError):
        f._fit()

def test_Fitter__unweighted_residuals(binding_curve_test_data):
    """
    Test unweighted residuals call against "manual" code used to generate
    test data. Just make sure answer is right; no error checking on this 
    function. 
    """

    # Build model
    df = binding_curve_test_data["df"]
    mw = ModelWrapper(binding_curve_test_data["wrappable_model"],
                      fittable_params=["K"])
    mw.df = df
    
    # Build fitter
    f = Fitter()
    f.model = mw
    f.y_obs = df.Y

    # Calculate residual given the input params
    input_params = binding_curve_test_data["input_params"]
    r = f._unweighted_residuals(input_params)

    assert np.allclose(r,df.residual)

def test_Fitter_unweighted_residuals(binding_curve_test_data):
    """
    Test unweighted residuals call against "manual" code used to generate
    test data.
    """

    df = binding_curve_test_data["df"]
    mw = ModelWrapper(binding_curve_test_data["wrappable_model"],
                      fittable_params=["K"])
    mw.df = df

    input_params = binding_curve_test_data["input_params"]
    
    f = Fitter()
    # Should fail, haven't loaded a model, y_obs or y_std yet
    with pytest.raises(RuntimeError):
        f.unweighted_residuals(input_params)

    f.model = mw

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
    mw = ModelWrapper(binding_curve_test_data["wrappable_model"],
                      fittable_params=["K"])
    mw.df = df
    
    # Build fitter
    f = Fitter()
    f.model = mw
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
    mw = ModelWrapper(binding_curve_test_data["wrappable_model"],
                      fittable_params=["K"])
    mw.df = df

    input_params = binding_curve_test_data["input_params"]
    
    f = Fitter()
    # Should fail, haven't loaded a model, y_obs or y_std yet
    with pytest.raises(RuntimeError):
        f.weighted_residuals(input_params)

    f.model = mw

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
    mw = ModelWrapper(binding_curve_test_data["wrappable_model"],
                      fittable_params=["K"])
    mw.df = df

    f = Fitter()
    f.model = mw
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
    mw = ModelWrapper(binding_curve_test_data["wrappable_model"],
                      fittable_params=["K"])
    mw.df = df

    input_params = binding_curve_test_data["input_params"]
    
    f = Fitter()
    # Should fail, haven't loaded a model, y_obs or y_std yet
    with pytest.raises(RuntimeError):
        f.ln_like(input_params)

    f.model = mw

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

def test_Fitter_model_setter_getter(binding_curve_test_data):
    """
    Test the model setter.
    """

    f = Fitter()

    def test_fcn(a=1,b=2,c="test"): return a*b
    
    # Not a function
    with pytest.raises(ValueError):
        f.model = "a"

    with pytest.raises(ValueError):
        f.model = test_fcn

    # Wrap standard model wrapper
    mw = ModelWrapper(test_fcn)
    f.model = mw
    assert f._model is mw
    assert f._fit_has_been_run is False

    assert f.model() == 1*2
    assert f.model([5,6]) == 5*6

    def test_fcn(a,c="test"): return a[0]*a[1]
    mw = VectorModelWrapper(model_to_fit=test_fcn,
                            fittable_params=["x","y"])
    mw.param_df["guess"] = [1,2]
    mw.finalize_params()

    # Wrap VectorModelWrapper
    f = Fitter()
    f.model = mw
    assert f._model is mw
    assert f._fit_has_been_run is False
    assert f.model() == 1*2
    assert f.model([10,20]) == 10*20

    # Already have a model. Cannot add a new one. 
    with pytest.raises(ValueError):
        f.model = mw

def test_Fitter_y_obs_setter_getter(binding_curve_test_data):
    """
    Test the y_obs setter.
    """

    f = Fitter()
 
    y_obs_input = np.array(binding_curve_test_data["df"].Y)

    f.y_obs = y_obs_input
    assert f.y_obs is not None
    assert np.array_equal(f.y_obs,y_obs_input)
    assert f._fit_has_been_run is False

    f = Fitter()
    with pytest.raises(ValueError):
        f.y_obs = "a"
    with pytest.raises(ValueError):
        f.y_obs = ["a","b"]

    # nan
    tmp_input = y_obs_input.copy()
    tmp_input[0] = np.nan
    with pytest.raises(ValueError):
        f.y_std = tmp_input

    f = Fitter()
    f.y_obs = y_obs_input
    assert np.array_equal(f.y_obs,y_obs_input)
    assert f.num_obs == y_obs_input.shape[0]

  
def test_Fitter_y_std_setter_getter(binding_curve_test_data):
    """
    Test the y_std setter.
    """

    y_obs_input = np.array(binding_curve_test_data["df"].Y)
    y_std_input = np.array(binding_curve_test_data["df"].Y_stdev)

    f = Fitter()
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

    f = Fitter()
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
    f = Fitter()
    assert f.y_std is None
    f.y_obs = y_obs_input
    f.y_std = 1.0
    assert np.array_equal(f.y_std,np.ones(f.y_obs.shape))


def test_Fitter_param_df():
    
    f = Fitter()
    assert f.param_df is None

    def test_fcn(a=1,b=2): return a*b
    mw = ModelWrapper(test_fcn)
    f.model = mw
    assert f.param_df is mw._param_df
    assert len(f.param_df) == 2
    assert np.array_equal(f.param_df["name"],["a","b"])


def test_Fitter_data_df():
    
    f = Fitter()
    out_df = f.data_df
    assert len(out_df) == 0
    
    y_obs = np.arange(10,dtype=float)
    y_std = np.ones(10,dtype=float)
    y_calc = np.arange(10)*0.9

    f = Fitter()
    f.y_obs = y_obs
    out_df = f.data_df
    assert len(out_df) == 10
    assert len(out_df.columns) == 1
    assert np.array_equal(out_df["y_obs"],y_obs)

    f = Fitter()
    f.y_obs = y_obs
    f.y_std = y_std
    out_df = f.data_df
    assert len(out_df) == 10
    assert len(out_df.columns) == 2
    assert np.array_equal(out_df["y_obs"],y_obs)
    assert np.array_equal(out_df["y_std"],y_std)
    
    f = Fitter()
    f.y_obs = y_obs
    f.y_std = y_std

    # hack it so it thinks its done
    f._success = True
    f._fit_df = {"estimate":[1,2]}
    def hack_fcn(a,b): return np.arange(10)*0.9
    mw = ModelWrapper(hack_fcn)
    f.model = mw
    f.model([2,3])

    # check final data_df
    out_df = f.data_df
    assert len(out_df) == 10
    assert len(out_df.columns) == 5
    assert np.array_equal(out_df["y_obs"],y_obs)
    assert np.array_equal(out_df["y_std"],y_std)
    assert np.array_equal(out_df["y_calc"],y_calc)
    assert np.array_equal(out_df["unweighted_residuals"],
                          y_obs - y_calc)
    assert np.array_equal(out_df["weighted_residuals"],
                          (y_obs - y_calc)/y_std)


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
    f = Fitter()
    with pytest.raises(NotImplementedError):
        f._update_fit_df()

def test_Fitter_fit_df():

    # This checks initialization. Need to write implementation-specific tests

    def test_fcn(a=1,b=2,x="array"): return x*a + b
    x = np.arange(10)
    y_obs = x*2 + 1
    mw = ModelWrapper(test_fcn)
    mw.x = x
    
    f = Fitter()
    assert f.fit_df is None

    f.model = mw
    assert len(f.fit_df) == 2
    assert np.array_equal(f.fit_df["name"],["a","b"])
    assert np.array_equal(f.fit_df.columns,
                          ["name","estimate","std","low_95","high_95",
                           "guess","fixed","lower_bound","upper_bound",
                           "prior_mean","prior_std"])

    f.y_obs = y_obs
    f.y_std = 0.1


def test_Fitter_samples():
    
    f = Fitter()
    assert f.samples is None
    f._samples = "something"
    assert f.samples == "something"


def test_Fitter_get_sample_df():
    
    # some test data
    y_obs = np.arange(10)
    y_std = np.ones(10)
    def test_fcn(a=1,b=2): return a*b*np.ones(10)
    mw = ModelWrapper(test_fcn)
    fake_samples = np.ones((1000,2),dtype=float)

    # Error checking on num_samples
    f = Fitter()
    with pytest.raises(ValueError):
        f.get_sample_df(num_samples=-1)
    with pytest.raises(ValueError):
        f.get_sample_df(num_samples="a")
        
    # empty class - return empty dataframe
    f = Fitter()
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
    f = Fitter()
    f.y_obs = y_obs
    f.y_std = y_std
    f.model = mw
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


def test_corner_plot():

    # tests run the whole decision tree of the function to identify major 
    # errors, but I'm not checking output here because it's graphical. 
    
    # some test data
    y_obs = np.arange(10)
    y_std = np.ones(10)
    def test_fcn(a=1,b=2): return a*b*np.ones(10)
    mw = ModelWrapper(test_fcn)
    fake_samples = np.random.normal(loc=0,scale=1,size=(1000,2))

    # Create a fitter that has apparently been run and has some samples
    f = Fitter()
    f.y_obs = y_obs
    f.y_std = y_std
    f.model = mw
    f._fit_df = pd.DataFrame({"name":["a","b"],"estimate":[10,20]})
    f._success = True
    f._samples = fake_samples

    # no fit_type specified
    fig = f.corner_plot()
    assert fig is None

    # set fit type, should now run
    f._fit_type = "fake"
    fig = f.corner_plot()
    assert issubclass(type(fig),matplotlib.figure.Figure)

    # Send in filter parameter possibilities. It should gracefully handle all
    # of these cases. 
    fig = f.corner_plot(filter_params=None)
    assert issubclass(type(fig),matplotlib.figure.Figure)
    
    fig = f.corner_plot(filter_params="blah")
    assert issubclass(type(fig),matplotlib.figure.Figure)

    fig = f.corner_plot(filter_params=["blah"])
    assert issubclass(type(fig),matplotlib.figure.Figure)

    fig = f.corner_plot(filter_params=[1])
    assert issubclass(type(fig),matplotlib.figure.Figure)
    
    # filter all parameters
    with pytest.raises(ValueError):
        fig = f.corner_plot(filter_params=["a","b"])
    
    # filter one
    fig = f.corner_plot(filter_params=["a"])
    assert issubclass(type(fig),matplotlib.figure.Figure)

    # filter other
    fig = f.corner_plot(filter_params=["b"])
    assert issubclass(type(fig),matplotlib.figure.Figure)

    # Get rid of samples attribute. Should now fail 
    f._samples = None
    with pytest.raises(RuntimeError):
        fig = f.corner_plot()

    # put samples back in
    f._samples = fake_samples
    fig = f.corner_plot(filter_params=None)
    assert issubclass(type(fig),matplotlib.figure.Figure)

    # pass in labels
    fig = f.corner_plot(filter_params=None,labels=["x","y"])
    assert issubclass(type(fig),matplotlib.figure.Figure)

    # pass in range
    fig = f.corner_plot(filter_params=None,range=[(-10,10),(-100,100)])
    assert issubclass(type(fig),matplotlib.figure.Figure)

    # pass in truths
    fig = f.corner_plot(filter_params=None,truths=[1,2])
    assert issubclass(type(fig),matplotlib.figure.Figure)


def test_Fitter_write_samples(tmpdir):
    
    cwd = os.getcwd()
    os.chdir(tmpdir)

    test_file = "test-out.pickle"

    # Should not write out because samples do not exist yet
    f = Fitter()
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
    mw = ModelWrapper(test_fcn)

    # Create fitter and assign model
    base_f = Fitter()
    base_f.model = mw

    # Assign samples
    base_f._samples = sample_array.copy()
    assert np.array_equal(base_f.samples.shape,(100,3))

    # add dummy function
    def dummy(*args,**kwargs): pass
    base_f._update_fit_df = dummy

    # -----------------------------------------------------------------------
    # Run tests

    # Nothing happens
    f = Fitter()
    f.append_samples(sample_file=None,
                     sample_array=None)
    
    # Check for existing samples (should fail without samples)
    f = Fitter()
    f.model = mw
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
    mw = ModelWrapper(test_fcn)
    f = Fitter()
    f.model = mw
    f.param_df.loc["a","fixed"] = True
    
    # some hacks to put this into a state to append samples
    f._samples = np.ones((100,2),dtype=float)
    f._update_fit_df = dummy

    with pytest.raises(ValueError):
        f.append_samples(sample_array=np.ones((100,3),dtype=float))
    
    f.append_samples(sample_array=np.ones((100,2),dtype=float))

    os.chdir(cwd)

def test_Fitter_num_params(binding_curve_test_data):

    f = Fitter()
    assert f.num_params is None

    def test_fcn(a=2,b=3): return a*b
    mw = ModelWrapper(test_fcn)
    f.model = mw

    assert f.num_params == 2

    assert f.model() == 2*3

    with pytest.raises(ValueError):
        f.model([7,8,9])

    f = Fitter()
    def test_fcn(a=2,b=3,c=4): return a*b*c
    mw = ModelWrapper(test_fcn)
    f.model = mw
    assert f.num_params == 3

    assert f.model() == 2*3*4

    with pytest.raises(ValueError):
        f.model([7,8,9,10])

def test_Fitter_num_obs():

    f = Fitter()
    assert f.num_obs is None

    f.y_obs = np.arange(10)
    assert f.num_obs == 10

    f = Fitter()
    f.y_obs = np.array([])
    assert f.num_obs == 0

def test_Fitter_num_unfixed_params():

    f = Fitter()
    assert f.num_unfixed_params is None
    
    def test_fcn(a=2,b=3,c=4): return a*b*c
    mw = ModelWrapper(test_fcn)
    f.model = mw
    
    assert f.num_unfixed_params == 3
    
    f.param_df.loc["a","fixed"] = True
    assert f.num_unfixed_params == 2

    f.param_df.loc["b","fixed"] = True
    assert f.num_unfixed_params == 1


def test_Fitter_fit_type():
    
    f = Fitter()
    assert f.fit_type == ""
    f._fit_type = "something"
    assert f.fit_type == "something"

def test_Fitter_success():
    
    f = Fitter()
    assert f.success is None
    f._success = True
    assert f.success is True

def test_Fitter_fit_info():
    
    f = Fitter()
    with pytest.raises(NotImplementedError):
        f.fit_info

def test_Fitter_fit_result():
    
    f = Fitter()
    assert f.fit_result is None
    f._fit_result = "something"
    assert f.fit_result == "something"
