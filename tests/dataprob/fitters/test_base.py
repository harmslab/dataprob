
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

    # now set _y_obs with setter
    f._y_obs = [1,2,3]
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
    with pytest.raises(ValueError):
        f._process_obs_args(y_obs=[1,2,3],
                            y_std=None)
    
    # ----------------------------------------------------------------------
    # y_std via setter, warn
    f = copy.deepcopy(f_base)
    assert f.y_obs is None
    assert f.y_std is None
    f._y_std = [1,1,1]

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



def test_Fitter_fit():

    def linear_fcn(m,b,x): return m*x + b
    x = np.linspace(-5,5,10)
    data_df = pd.DataFrame({"y_obs":linear_fcn(m=2,b=-1,x=x),
                            "y_std":0.1*np.ones(10)})

    base_kwargs = {"y_obs":data_df.y_obs,
                   "y_std":data_df.y_std,
                   "fit_kwarg":5}

    def new_fitter():

        # Create a fitter with a model, then hacked _fit, _fit_result, 
        # and _update_fit_df
        f = Fitter(some_function=linear_fcn)
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
    kwargs["y_obs"] = data_df["y_obs"]
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

def test_Fitter__unweighted_residuals():
    """
    Test unweighted residuals call with linear function.
    """

    test_params =  np.array([10,20])
    def linear_fcn(m,b,x): return m*x + b
    x = np.linspace(-5,5,15)
    f = Fitter(some_function=linear_fcn,
               non_fit_kwargs={"x":x})
    
    f.data_df = pd.DataFrame({"y_obs":linear_fcn(m=2,b=-1,x=x),
                              "y_std":0.1*np.ones(15)})

    assert np.allclose(linear_fcn(10,20,x) - linear_fcn(m=2,b=-1,x=x),
                       f._unweighted_residuals(test_params))


def test_Fitter_unweighted_residuals():
    """
    Test unweighted residuals like _unweighted_residuals, but test error 
    checking. 
    """

    test_params =  np.array([10,20])
    def linear_fcn(m,b,x): return m*x + b
    x = np.linspace(-5,5,15)
    f = Fitter(some_function=linear_fcn,
               non_fit_kwargs={"x":x})

    # Should fail, haven't loaded y_obs or y_std yet
    with pytest.raises(RuntimeError):
        f.unweighted_residuals(test_params)

    # Load in y_obs and y_std
    f.data_df = pd.DataFrame({"y_obs":linear_fcn(m=2,b=-1,x=x),
                              "y_std":0.1*np.ones(15)})

    # Should work now
    assert np.allclose(linear_fcn(10,20,x) - linear_fcn(m=2,b=-1,x=x),
                       f.unweighted_residuals(test_params))

    # Make sure error check is running by sending in too many parameters
    with pytest.raises(ValueError):
        f.unweighted_residuals([1,2,3,4])

def test_Fitter__weighted_residuals():
    """
    Test weighted residuals call with linear function.
    """

    test_params =  np.array([10,20])
    def linear_fcn(m,b,x): return m*x + b
    x = np.linspace(-5,5,15)
    f = Fitter(some_function=linear_fcn,
               non_fit_kwargs={"x":x})
    
    f.data_df = pd.DataFrame({"y_obs":linear_fcn(m=2,b=-1,x=x),
                              "y_std":0.1*np.ones(15)})

    assert np.allclose((linear_fcn(10,20,x) - linear_fcn(m=2,b=-1,x=x))/0.1,
                       f._weighted_residuals(test_params))


def test_Fitter_weighted_residuals():
    """
    Test weighted residuals like _weighted_residuals, but test error 
    checking. 
    """

    test_params =  np.array([10,20])
    def linear_fcn(m,b,x): return m*x + b
    x = np.linspace(-5,5,15)
    f = Fitter(some_function=linear_fcn,
               non_fit_kwargs={"x":x})

    # Should fail, haven't loaded y_obs or y_std yet
    with pytest.raises(RuntimeError):
        f.weighted_residuals(test_params)

    # Load in y_obs and y_std
    f.data_df = pd.DataFrame({"y_obs":linear_fcn(m=2,b=-1,x=x),
                              "y_std":0.1*np.ones(15)})

    # Should work now
    assert np.allclose((linear_fcn(10,20,x) - linear_fcn(m=2,b=-1,x=x))/0.1,
                       f.weighted_residuals(test_params))

    # Make sure error check is running by sending in too many parameters
    with pytest.raises(ValueError):
        f.weighted_residuals([1,2,3,4])

def test_Fitter__ln_like():
    """
    Test internal function -- no error checking. 
    """

    def linear_fcn(m,b,x): return m*x + b
    x = np.linspace(-5,5,15)
    y_obs = linear_fcn(m=2,b=-1,x=x)
    y_std = 0.1*np.ones(15)
    y_calc = linear_fcn(m=10,b=20,x=x)
    test_params = np.array([10,20])

    f = Fitter(some_function=linear_fcn,
               non_fit_kwargs={"x":x})
    f.data_df = pd.DataFrame({"y_obs":y_obs,
                              "y_std":y_std})

    sigma2 = y_std**2
    ln_like = -0.5*(np.sum((y_calc - y_obs)**2/sigma2 + np.log(2*np.pi*sigma2)))

    assert np.isclose(f._ln_like(test_params),ln_like)

def test_Fitter_ln_like():
    """
    Test ln_like like _ln_like, but test error checking. 
    """

    def linear_fcn(m,b,x): return m*x + b
    x = np.linspace(-5,5,15)
    y_obs = linear_fcn(m=2,b=-1,x=x)
    y_std = 0.1*np.ones(15)
    y_calc = linear_fcn(m=10,b=20,x=x)
    test_params = np.array([10,20])

    f = Fitter(some_function=linear_fcn,
               non_fit_kwargs={"x":x})
    
    # Should fail, haven't loaded y_obs or y_std yet
    with pytest.raises(RuntimeError):
        f.ln_like(test_params)

    f.data_df = pd.DataFrame({"y_obs":y_obs,
                              "y_std":y_std})

    # should work now
    sigma2 = y_std**2
    ln_like = -0.5*(np.sum((y_calc - y_obs)**2/sigma2 + np.log(2*np.pi*sigma2)))

    assert np.isclose(f._ln_like(test_params),ln_like)
    
    # make sure input params sanity check is running
    with pytest.raises(ValueError):
        f.ln_like([1,2,3,4])


# ---------------------------------------------------------------------------- #
# Test setters, getters, and internal sanity checks
# ---------------------------------------------------------------------------- #

def test_Fitter_y_obs():
    """
    Test the y_obs setter.
    """

    def test_fcn(x=1,y=2): return x*y
    f = Fitter(some_function=test_fcn)
    assert f.y_obs is None
    f._y_obs = "something"
    assert f.y_obs == "something"
    

  
def test_Fitter_y_std():
    """
    Test the y_std setter.
    """

    def test_fcn(x=1,y=2): return x*y
    f = Fitter(some_function=test_fcn)
    assert f.y_std is None
    f._y_std = "something"
    assert f.y_std == "something"


def test_Fitter_param_df():
    
    def test_fcn(a=1,b=2): return a*b
    f = Fitter(some_function=test_fcn)
    
    assert len(f.param_df) == 2
    assert np.array_equal(f.param_df["name"],["a","b"])

    not_good = pd.DataFrame({'name':["x","y"]})
    with pytest.raises(ValueError):
        f.param_df = not_good
    
    assert np.array_equal(f.param_df["guess"],[1,2])
    good_df = pd.DataFrame({"name":["a","b"],"guess":[3,4]})
    f.param_df = good_df
    assert np.array_equal(f.param_df["guess"],[3,4])
    

        
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
    
    # test getter
    
    def test_fcn(a=1,b=2): return a*b
    f = Fitter(some_function=test_fcn)

    out_df = f.data_df
    assert len(out_df) == 0

    y_obs = np.arange(10,dtype=float)
    y_std = np.ones(10,dtype=float)

    f = Fitter(some_function=test_fcn)
    f._y_obs = y_obs
    out_df = f.data_df
    assert len(out_df) == 10
    assert len(out_df.columns) == 1
    assert np.array_equal(out_df["y_obs"],y_obs)

    f = Fitter(some_function=test_fcn)
    f._y_obs = y_obs
    f._y_std = y_std
    out_df = f.data_df
    assert len(out_df) == 10
    assert len(out_df.columns) == 2
    assert np.array_equal(out_df["y_obs"],y_obs)
    assert np.array_equal(out_df["y_std"],y_std)
    
    # Create real function and fitter
    def linear_fcn(m=1,b=2,x=None): return m*x + b
    f = Fitter(some_function=linear_fcn,
               non_fit_kwargs={"x":np.arange(10)})
    y_obs = 1*np.arange(10) + 2
    y_std = np.ones(10,dtype=float)*0.1
    y_calc = 1*np.arange(10) + 3
    f._y_obs = y_obs
    f._y_std = y_std
    f._success = True
    f._fit_df = pd.DataFrame({"estimate":[1,3]})
    f._model._unfixed_mask = np.ones(2,dtype=bool)

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

    # set setter
    def test_fcn(a=1,b=2): return a*b
    f = Fitter(some_function=test_fcn)
    f._fit_has_been_run = True # hack to True to check that it gets set to F

    tmp_df = pd.DataFrame({"y_obs":[1,2],
                           "y_std":[3,4]})
    f.data_df = tmp_df
    assert np.array_equal(f._y_obs,[1,2])
    assert np.array_equal(f._y_std,[3,4])
    assert np.array_equal(f.data_df["y_obs"],[1,2])
    assert np.array_equal(f.data_df["y_std"],[3,4])
    assert not f._fit_has_been_run

    # missing column
    f = Fitter(some_function=test_fcn)
    tmp_df = pd.DataFrame({"y_obs":[1,2]})
    with pytest.raises(ValueError):
        f.data_df = tmp_df
    
    # missing column
    f = Fitter(some_function=test_fcn)
    tmp_df = pd.DataFrame({"y_std":[1,2]})
    with pytest.raises(ValueError):
        f.data_df = tmp_df

    # bad column name
    f = Fitter(some_function=test_fcn)
    tmp_df = pd.DataFrame({"y_obs_y":[1,2],
                           "y_std":[3,4]})
    with pytest.raises(ValueError):
        f.data_df = tmp_df

    # non-numeric column
    f = Fitter(some_function=test_fcn)
    tmp_df = pd.DataFrame({"y_obs":["not",2],
                           "y_std":[3,4]})
    with pytest.raises(ValueError):
        f.data_df = tmp_df

    # nan column
    f = Fitter(some_function=test_fcn)
    tmp_df = pd.DataFrame({"y_obs":[np.nan,2],
                           "y_std":[3,4]})
    with pytest.raises(ValueError):
        f.data_df = tmp_df
    
    # inf column
    f = Fitter(some_function=test_fcn)
    tmp_df = pd.DataFrame({"y_obs":[np.inf,2],
                           "y_std":[3,4]})
    with pytest.raises(ValueError):
        f.data_df = tmp_df

    # bad std
    f = Fitter(some_function=test_fcn)
    tmp_df = pd.DataFrame({"y_obs":[1,2],
                           "y_std":[0,4]})
    with pytest.raises(ValueError):
        f.data_df = tmp_df

    
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

def test_Fitter_fit_quality():

    def test_fcn(a=1,b=2,x="array"): return x*a + b
    x = np.arange(10)
    
    f = Fitter(some_function=test_fcn,
               non_fit_kwargs={"x":x})

    # nothing done yet
    assert f.fit_quality is None



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
    f._y_obs = y_obs
    sample_df = f.get_sample_df()
    assert issubclass(type(sample_df),pd.DataFrame)
    assert len(sample_df) == 10
    assert np.array_equal(sample_df["y_obs"],y_obs)
    assert np.array_equal(sample_df.columns,["y_obs"])

    # add y_std, should now be in dataframe
    f._y_std = y_std
    sample_df = f.get_sample_df()
    assert issubclass(type(sample_df),pd.DataFrame)
    assert len(sample_df) == 10
    assert np.array_equal(sample_df["y_obs"],y_obs)
    assert np.array_equal(sample_df["y_std"],y_std)
    assert np.array_equal(sample_df.columns,["y_obs","y_std"])

    # Create a fitter that has apparently been run, but has no samples
    f = Fitter(some_function=test_fcn)
    f._y_obs = y_obs
    f._y_std = y_std
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

def test_Fitter_num_params():

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

    f._y_obs = np.arange(10)
    assert f.num_obs == 10

    f = Fitter(some_function=test_fcn)
    f._y_obs = np.array([])
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
