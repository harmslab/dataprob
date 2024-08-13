
import pytest

from dataprob.fitters.base import Fitter
from dataprob.model_wrapper.model_wrapper import ModelWrapper
from dataprob.model_wrapper.vector_model_wrapper import VectorModelWrapper

import numpy as np
import pandas as pd

import os
import pickle
import copy

# ---------------------------------------------------------------------------- #
# Test __init__ 
# ---------------------------------------------------------------------------- #

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

def test_Fitter_reconcile_model_args():

    # Create a fitter that already has a model
    f = Fitter()
    def test_fcn(a=1,b=2): return a*b
    mw = ModelWrapper(model_to_fit=test_fcn)
    f.model = mw

    # Should run. 
    f._reconcile_model_args(model=None,guesses=None,names=None)
    assert f._model is mw

    # Die. Cannot specify a new model or names
    with pytest.raises(ValueError):
        f._reconcile_model_args(model=test_fcn,guesses=[1,2],names=["a","b"])

    # Die. Cannot specify a new model
    with pytest.raises(ValueError):
        f._reconcile_model_args(model=test_fcn,guesses=None,names=None)

    # Die. Cannot specify a new names
    with pytest.raises(ValueError):
        f._reconcile_model_args(model=None,guesses=None,names=["a","b"])

    # Create an empty fitter
    f = Fitter()
    with pytest.raises(AttributeError):
        f._model

    # No model sent in, die.
    with pytest.raises(ValueError):
        f._reconcile_model_args(model=None,guesses=[1,2],names=["a","b"])

    # Extra names sent in. Die
    with pytest.raises(ValueError):
        f._reconcile_model_args(model=mw,guesses=[1,2],names=["a","b"])
    
    # model and guesses -- fine
    f._reconcile_model_args(model=mw,guesses=[1,2],names=None)    
    assert f._model is mw

    # Model and no guesses, fine. 
    f = Fitter()
    with pytest.raises(AttributeError):
        f._model
    f._reconcile_model_args(model=mw,guesses=None,names=None)    
    assert f._model is mw

    # Send in naked function, a is a length-two list
    def test_fcn(a,b=2): return a[0]*a[1]*b
    f = Fitter()
    with pytest.raises(AttributeError):
        f._model
    
    # Naked function. Die because no guesses or names specified. 
    with pytest.raises(ValueError):
        f._reconcile_model_args(model=test_fcn,guesses=None,names=None)
    
    # Naked function. Die because no guesses or names specified. 
    with pytest.raises(ValueError):
        f._reconcile_model_args(model=test_fcn,guesses=None,names=None)
    
    # Naked function. Die because no guesses  specified. 
    with pytest.raises(ValueError):
        f._reconcile_model_args(model=test_fcn,guesses=None,names=["x","y"])

    # Naked function. Work. 
    f._reconcile_model_args(model=test_fcn,guesses=[5,6],names=["x","y"])
    np.array_equal(f._model.param_df["name"],["x","y"])

    # Naked function. Work and create default parameter names
    f = Fitter()
    f._reconcile_model_args(model=test_fcn,guesses=[5,6],names=None)
    np.array_equal(f._model.param_df["name"],["p0","p1"])

    # Naked function. die because guesses and names have different lengths
    f = Fitter()
    with pytest.raises(ValueError):
        f._reconcile_model_args(model=test_fcn,guesses=[5,6],names=["x","y","z"])


def xtest_Fitter_fit(fitter_object,binding_curve_test_data):
    
    def dummy_fit(f,N,*args,**kwargs):
        """
        This function takes f and N and uses that to set fit results without
        actually doing anything. It should be invoked by
        
        f = Fitter()
        f._fit = dummy_fit
        
        then 
        
        f.fit(f=f,N=N)

        f and N are passed to dummy fit, which updates the fitter attributes 
        appropriately fro the test. 
        """
        f._fit_result = {}
        f._success = True
        
        f._estimate = np.zeros(N,dtype=float)
        f._stdev = 0.5*np.ones(N,dtype=float)
        f._ninetyfive = 1.0*np.ones((2,N),dtype=float)

    N = len(binding_curve_test_data["guesses"])
    kwargs = {"N":N,
              "model":binding_curve_test_data["generic_model"],
              "y_obs":binding_curve_test_data["df"].Y,
              "y_std":binding_curve_test_data["df"].y_std,
              "guesses":[5],
              "names":["blah"],
              "priors":[[0],[10]],
              "bounds":[[-100],[100]]}

    # Send in a generic model that will make us specify everything, and make 
    # sure specifications are working
    f = Fitter()
    f._fit = dummy_fit
    
    test_kwargs = copy.deepcopy(kwargs)

    f.fit(f=f,**test_kwargs)
    
    assert np.array_equal(f.y_obs,binding_curve_test_data["df"].Y)
    assert np.array_equal(f.y_std,binding_curve_test_data["df"].y_std)
    assert np.array_equal(f.guesses,[5])
    assert np.array_equal(f.priors,[[0],[10]])
    assert np.array_equal(f.names,["blah"])
    assert f._fit_has_been_run is True

    # Send in a generic model with no y_std. Make sure it warns and sets
    # value correctly. 
    f = Fitter()
    f._fit = dummy_fit
    
    test_kwargs = copy.deepcopy(kwargs)
    test_kwargs["y_std"] = None
    with pytest.warns():
        f.fit(f=f,**test_kwargs)
    
    scalar = np.mean(np.abs(binding_curve_test_data["df"].Y))*0.1
    stdev = scalar*np.ones(len(binding_curve_test_data["df"].Y))

    assert np.array_equal(f.y_obs,binding_curve_test_data["df"].Y)
    assert np.allclose(f.y_std,stdev)
    assert np.array_equal(f.guesses,[5])
    assert np.array_equal(f.priors,[[0],[10]])
    assert np.array_equal(f.names,["blah"])
    assert f._fit_has_been_run is True

    # Send in a generic model that will make us specify everything, but send in
    # badness for each and make sure it throws error.
    f = Fitter()
    f._fit = dummy_fit
    test_kwargs = copy.deepcopy(kwargs)
    test_kwargs["model"] = "not_callable"
    with pytest.raises(ValueError):
        f.fit(f=f,**test_kwargs)
    
    f = Fitter()
    f._fit = dummy_fit
    test_kwargs = copy.deepcopy(kwargs)
    test_kwargs["y_obs"] = "not_yobs"
    with pytest.raises(ValueError):
        f.fit(f=f,**test_kwargs)
    
    f = Fitter()
    f._fit = dummy_fit
    test_kwargs = copy.deepcopy(kwargs)
    test_kwargs["y_std"] = "not_stdev"
    with pytest.raises(ValueError):
        f.fit(f=f,**test_kwargs)
    
    f = Fitter()
    f._fit = dummy_fit
    test_kwargs = copy.deepcopy(kwargs)
    test_kwargs["guesses"] = [1,2,3]
    with pytest.raises(ValueError):
        f.fit(f=f,**test_kwargs)

    f = Fitter()
    f._fit = dummy_fit
    test_kwargs = copy.deepcopy(kwargs)
    test_kwargs["names"] = ["a","b"]
    with pytest.raises(ValueError):
        f.fit(f=f,**test_kwargs)

    f = Fitter()
    f._fit = dummy_fit
    test_kwargs = copy.deepcopy(kwargs)
    test_kwargs["priors"] = "not_prior"
    with pytest.raises(ValueError):
        f.fit(f=f,**test_kwargs)

    f = Fitter()
    f._fit = dummy_fit
    test_kwargs = copy.deepcopy(kwargs)
    test_kwargs["bounds"] = "not_bounds"
    with pytest.raises(ValueError):
        f.fit(f=f,**test_kwargs)
    
    # Default run should fail because model is not specified
    f = Fitter()
    f._fit = dummy_fit
    with pytest.raises(RuntimeError):
        f.fit(f=f,N=N)

    # Send in an unwrapped model. Should fail because no guesses. 
    f = Fitter()
    f._fit = dummy_fit
    f.model = binding_curve_test_data["generic_model"]
    with pytest.raises(RuntimeError):
        f.fit(f=f,N=N)

    # Default run will work with wrapped model because it will bring in all 
    # values but will throw a warning because stdev are made up
    f = Fitter()
    f._fit = dummy_fit
    mw = ModelWrapper(binding_curve_test_data["wrappable_model"])
    f.model = mw
    f.y_obs = binding_curve_test_data["df"].Y
    with pytest.warns():
        f.fit(f=f,N=N)

    scalar = np.mean(np.abs(f.y_obs))*0.1

    assert np.array_equal(f.guesses,np.ones(N))
    assert np.array_equal(f.priors,np.nan*np.ones((2,N)),equal_nan=True)
    assert np.array_equal(f.names,["K"])
    assert np.allclose(f.y_std,scalar*np.ones(len(f.y_obs)))
    assert f._fit_has_been_run is True

    # Send in a generic model that will make us pre-specify many features 
    f = Fitter()
    f._fit = dummy_fit
    f.model = binding_curve_test_data["generic_model"]
    f.y_obs = binding_curve_test_data["df"].Y
    f.y_std = binding_curve_test_data["df"].y_std
    with pytest.raises(RuntimeError):
        f.fit(f=f,N=N)

    # works with guesses sent in
    f.fit(f=f,
          N=N,
          guesses=[0])
    
    assert np.array_equal(f.guesses,np.zeros(N))
    assert np.array_equal(f.priors,np.nan*np.ones((2,N)),equal_nan=True)
    assert np.array_equal(f.names,["p0"])
    assert np.array_equal(f.y_std,binding_curve_test_data["df"].y_std)
    assert f._fit_has_been_run is True

    # fix all parameters. should now fail because nothing is floating
    f = Fitter()
    f._fit = dummy_fit
    mw = ModelWrapper(binding_curve_test_data["wrappable_model"])
    for p in mw.fit_parameters:
        mw.fit_parameters[p].fixed = True
    f.model = mw
    f.y_obs = binding_curve_test_data["df"].Y
    f.y_std = binding_curve_test_data["df"].y_std
    with pytest.raises(RuntimeError):
        f.fit(f=f,N=N)

def test_Fitness__fit():
    f = Fitter()
    with pytest.raises(NotImplementedError):
        f._fit()

def test_Fitness__unweighted_residuals(binding_curve_test_data):
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

def test_Fitness_unweighted_residuals(binding_curve_test_data):
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

def test_Fitness__weighted_residuals(binding_curve_test_data):
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


def test_Fitness_weighted_residuals(binding_curve_test_data):
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

def test_Fitness__ln_like(binding_curve_test_data):
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

def test_Fitness_ln_like(binding_curve_test_data):
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

def test_Fitness_model_setter_getter(binding_curve_test_data):
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

def test_Fitness_y_obs_setter_getter(binding_curve_test_data):
    """
    Test the y_obs setter.
    """

    f = Fitter()
 
    f.y_obs = binding_curve_test_data["df"].Y
    assert f.y_obs is not None
    assert np.array_equal(f.y_obs,binding_curve_test_data["df"].Y)
    assert f._fit_has_been_run is False

    f = Fitter()
    with pytest.raises(ValueError):
        f.y_obs = "a"
    with pytest.raises(ValueError):
        f.y_obs = ["a","b"]

    f = Fitter()
    input_data = np.array(binding_curve_test_data["df"].Y)
    f.y_obs = input_data
    assert np.array_equal(f.y_obs,input_data)
    assert f.num_obs == input_data.shape[0]

  
def test_Fitness_y_std_setter_getter(binding_curve_test_data):
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
    
    f.y_std = y_std_input
    assert np.array_equal(y_std_input,f.y_std)
    assert f._fit_has_been_run is False

    # Set single value
    f = Fitter()
    assert f.y_std is None
    f.y_obs = y_obs_input
    f.y_std = 1.0
    assert np.array_equal(f.y_std,np.ones(f.y_obs.shape))


def test_Fitness_param_df():
    
    f = Fitter()
    assert f.param_df is None

    def test_fcn(a=1,b=2): return a*b
    mw = ModelWrapper(test_fcn)
    f.model = mw
    assert f.param_df is mw._param_df
    assert len(f.param_df) == 2
    assert np.array_equal(f.param_df["name"],["a","b"])


def test_Fitness_data_df():
    
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


def test_Fitness__initialize_fit_df():
    
    # test on fake class
    class TestClass:
        def __init__(self):
            self.param_df = {"name":["a","b"]}
    
    tc = TestClass()
    Fitter._initialize_fit_df(tc)
    assert np.array_equal(tc._fit_df["name"],["a","b"])
    assert np.sum(np.isnan(tc._fit_df["estimate"]))
    assert np.sum(np.isnan(tc._fit_df["std"]))
    assert np.sum(np.isnan(tc._fit_df["low_95"]))
    assert np.sum(np.isnan(tc._fit_df["high_95"]))
    

def test_Fitness__update_fit_df():
    f = Fitter()
    with pytest.raises(NotImplementedError):
        f._update_fit_df()

def test_Fitness_fit_df():

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
                          ["name","estimate","std","low_95","high_95"])

    f.y_obs = y_obs
    f.y_std = 0.1


def xtest_Fitness_samples():
    pass


def xtest_Fitness_get_sample_df():
    pass


def xtest_corner_plot():
    ## MOVE THIS FUNCTION
    pass

def test_Fitness_write_samples(tmpdir):
    
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

def test_Fitness_append_samples(tmpdir):

    cwd = os.getcwd()
    os.chdir(tmpdir)

    # make some files and arrays for testing
    sample_array = np.ones((100,3),dtype=float)
    with open("test.pickle","wb") as p:
        pickle.dump(sample_array,p)
    with open("bad_file.txt","w") as g:
        g.write("yo")

    # Build a hacked Fitter object that has existing samples, three params, 
    # and an overwritten _update_estimates call that does nothing.
    base_f = Fitter()
    base_f._samples = sample_array.copy()
    base_f._num_params = 3
    assert np.array_equal(base_f.samples.shape,(100,3))
    def dummy(*args,**kwargs): pass
    base_f._update_estimates = dummy

    f = Fitter()

    # Nothing happens
    f.append_samples(sample_file=None,
                     sample_array=None)
    
    # Check for existing samples (should fail without samples)
    f = Fitter()
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
    f._update_estimates = dummy
    
    with pytest.raises(RuntimeError):
        f.append_samples(sample_array=sample_array)

    os.chdir(cwd)

def test_Fitness_num_params(binding_curve_test_data):

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

def test_Fitness_num_obs():

    f = Fitter()
    assert f.num_obs is None

    f.y_obs = np.arange(10)
    assert f.num_obs == 10

    f = Fitter()
    f.y_obs = np.array([])
    assert f.num_obs == 0

def xtest_Fitness_fit_type():
    pass

def xtest_Fitness_success():
    pass

def xtest_Fitness_fit_info():
    pass

def xtest_Fitness_fit_result():
    pass
