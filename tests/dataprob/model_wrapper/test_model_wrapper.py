import pytest

from dataprob.model_wrapper.model_wrapper import ModelWrapper
from dataprob.fit_param import FitParameter

import numpy as np
import pandas as pd


# ----- _mw_observable, _update_parameter_map, __getattr__, __setattr__, _mw_load_model, __init__

def test_load_fit_result(binding_curve_test_data,fitter_object):

    model_to_fit = binding_curve_test_data["wrappable_model"]
    previous_fit = fitter_object["wrapped_fit"]

    mw = ModelWrapper(model_to_fit)
    assert mw.K.value == mw.K.guess
    assert mw.K.stdev is None
    assert mw.K.is_fit_result is False

    mw.load_fit_result(previous_fit)
    assert mw.K.value != mw.K.guess
    assert mw.K.value == previous_fit.estimate[0]
    assert mw.K.stdev == previous_fit.stdev[0]
    assert mw.K.is_fit_result is True

    # Parameter mismatch -- should throw error
    def test_model(K=20,a=3): return K*a
    mw = ModelWrapper(test_model)
    with pytest.raises(ValueError):
        mw.load_fit_result(previous_fit)


def test_ModelWrapper_load_param_spreadsheet():

    # Simple model
    def model_to_test_wrap(a=1,b=2,c=3,d="test",e=3): return a*b*c
    
    # ---------------------------------------------------------------
    # guesses

    mw = ModelWrapper(model_to_test_wrap)
    assert np.array_equal(mw.guesses,[1,2,3])
    assert mw.model() == 1*2*3

    # Load value from spreadsheet
    df = pd.DataFrame({"param":["a","b","c"],
                       "guess":[10,20,30]})
    mw.load_param_spreadsheet(spreadsheet=df)
    assert np.array_equal(mw.guesses,[10,20,30])
    assert mw.model() == 10*20*30

    # bad value
    df = pd.DataFrame({"param":["a","b","c"],
                       "guess":["x","y","z"]})
    with pytest.raises(ValueError):
        mw.load_param_spreadsheet(spreadsheet=df)
    
    # ---------------------------------------------------------------
    # fixed

    mw = ModelWrapper(model_to_test_wrap)
    assert np.array_equal(mw.fixed_mask,[False,False,False])

    # Load value from spreadsheet
    df = pd.DataFrame({"param":["a","b","c"],
                       "fixed":[True,False,True]})
    mw.load_param_spreadsheet(spreadsheet=df)
    assert np.array_equal(mw.fixed_mask,[True,False,True])

    # bad value
    df = pd.DataFrame({"param":["a","b","c"],
                       "fixed":["x","y","z"]})
    with pytest.raises(ValueError):
        mw.load_param_spreadsheet(spreadsheet=df)

    # ---------------------------------------------------------------
    # lower_bound

    mw = ModelWrapper(model_to_test_wrap)
    assert np.array_equal(mw.bounds[0],[-np.inf,-np.inf,-np.inf])

    # Load value from spreadsheet
    df = pd.DataFrame({"param":["a","b","c"],
                       "lower_bound":[1,2,3]})
    mw.load_param_spreadsheet(spreadsheet=df)
    assert np.array_equal(mw.bounds[0],[1,2,3])

    # bad value
    df = pd.DataFrame({"param":["a","b","c"],
                       "lower_bound":["x","y","z"]})
    with pytest.raises(ValueError):
        mw.load_param_spreadsheet(spreadsheet=df)
    
    # ---------------------------------------------------------------
    # upper_bound

    mw = ModelWrapper(model_to_test_wrap)
    assert np.array_equal(mw.bounds[1],[np.inf,np.inf,np.inf])

    # Load value from spreadsheet
    df = pd.DataFrame({"param":["a","b","c"],
                       "upper_bound":[1,2,3]})
    mw.load_param_spreadsheet(spreadsheet=df)
    assert np.array_equal(mw.bounds[1],[1,2,3])

    # bad value
    df = pd.DataFrame({"param":["a","b","c"],
                       "upper_bound":["x","y","z"]})
    with pytest.raises(ValueError):
        mw.load_param_spreadsheet(spreadsheet=df)

    # ---------------------------------------------------------------
    # lower and upper_bound

    mw = ModelWrapper(model_to_test_wrap)
    assert np.array_equal(mw.bounds[0],[-np.inf,-np.inf,-np.inf])
    assert np.array_equal(mw.bounds[1],[np.inf,np.inf,np.inf])

    # Load value from spreadsheet
    df = pd.DataFrame({"param":["a","b","c"],
                       "lower_bound":[-1,-2,-3],
                       "upper_bound":[1,2,3]})
    mw.load_param_spreadsheet(spreadsheet=df)
    assert np.array_equal(mw.bounds[0],[-1,-2,-3])
    assert np.array_equal(mw.bounds[1],[1,2,3])

    # bad value
    df = pd.DataFrame({"param":["a","b","c"],
                       "lower_bound":[-1,-2,-3],
                       "upper_bound":["x","y","z"]})
    with pytest.raises(ValueError):
        mw.load_param_spreadsheet(spreadsheet=df)
    
    # more interesting bad value where the bounds are flipped
    df = pd.DataFrame({"param":["a","b","c"],
                       "upper_bound":[-1,-2,-3],
                       "lower_bound":[1,2,3]})
    with pytest.raises(ValueError):
        mw.load_param_spreadsheet(spreadsheet=df)

    # ---------------------------------------------------------------
    # prior

    mw = ModelWrapper(model_to_test_wrap)
    assert np.array_equal(mw.priors[0],[np.nan,np.nan,np.nan],equal_nan=True)
    assert np.array_equal(mw.priors[1],[np.nan,np.nan,np.nan],equal_nan=True)

    # Load value from spreadsheet
    df = pd.DataFrame({"param":["a","b","c"],
                       "prior_mean":[-1,-2,-3],
                       "prior_std":[1,2,3]})
    mw.load_param_spreadsheet(spreadsheet=df)
    assert np.array_equal(mw.priors[0],[-1,-2,-3])
    assert np.array_equal(mw.priors[1],[1,2,3])

    # bad value
    df = pd.DataFrame({"param":["a","b","c"],
                       "prior_mean":[-1,-2,-3],
                       "prior_std":["x","y","z"]})
    with pytest.raises(ValueError):
        mw.load_param_spreadsheet(spreadsheet=df)
    
    # missing std
    df = pd.DataFrame({"param":["a","b","c"],
                       "prior_mean":[-1,-2,-3]})
    with pytest.raises(ValueError):
        mw.load_param_spreadsheet(spreadsheet=df)

    # missing mean
    df = pd.DataFrame({"param":["a","b","c"],
                       "prior_std":[1,2,3]})
    with pytest.raises(ValueError):
        mw.load_param_spreadsheet(spreadsheet=df)
    
    df = pd.DataFrame({"param":["a","b","c"],
                       "prior_mean":[1,2,np.nan],
                       "prior_std":[1,2,3]})
    with pytest.raises(ValueError):
        mw.load_param_spreadsheet(spreadsheet=df)
    
    # invalid std (negative)
    df = pd.DataFrame({"param":["a","b","c"],
                       "prior_mean":[-1,-2,-3],
                       "prior_std":[1,2,-3]})
    with pytest.raises(ValueError):
        mw.load_param_spreadsheet(spreadsheet=df)

    # both missing; okay
    mw = ModelWrapper(model_to_test_wrap)
    assert np.array_equal(mw.priors[0],[np.nan,np.nan,np.nan],equal_nan=True)
    assert np.array_equal(mw.priors[1],[np.nan,np.nan,np.nan],equal_nan=True)

    df = pd.DataFrame({"param":["a","b","c"],
                       "prior_mean":[-1,-2,None],
                       "prior_std":[1,2,None]})
    mw.load_param_spreadsheet(spreadsheet=df)
    assert np.array_equal(np.isnan(mw.priors[0]),[False,False,True])
    assert np.array_equal(np.isnan(mw.priors[1]),[False,False,True])
    assert np.array_equal(mw.priors[0,:2],[-1,-2])
    assert np.array_equal(mw.priors[1,:2],[1,2])

    # ---------------------------------------------------------------
    # Set all for one parameter

    mw = ModelWrapper(model_to_test_wrap)
    assert np.array_equal(mw.guesses,[1,2,3])
    assert np.array_equal(mw.fixed_mask,[False,False,False])
    assert np.array_equal(mw.bounds[0],[-np.inf,-np.inf,-np.inf])
    assert np.array_equal(mw.bounds[1],[np.inf,np.inf,np.inf])
    assert np.array_equal(mw.priors[0],[np.nan,np.nan,np.nan],equal_nan=True)
    assert np.array_equal(mw.priors[1],[np.nan,np.nan,np.nan],equal_nan=True)

    df = pd.DataFrame({"param":["a"],
                       "guess":[5],
                       "lower_bound":[-10],
                       "upper_bound":[10],
                       "prior_mean":[-1],
                       "prior_std":[1]})
    mw.load_param_spreadsheet(spreadsheet=df)
    assert np.array_equal(mw.guesses,[5,2,3])
    assert np.array_equal(mw.fixed_mask,[False,False,False])
    assert np.array_equal(mw.bounds[0],[-10,-np.inf,-np.inf])
    assert np.array_equal(mw.bounds[1],[10,np.inf,np.inf])
    assert np.array_equal(mw.priors[0],[-1,np.nan,np.nan],equal_nan=True)
    assert np.array_equal(mw.priors[1],[1,np.nan,np.nan],equal_nan=True)
    
    # ---------------------------------------------------------------
    # Fix single parameter (continued from last)

    df = pd.DataFrame({"param":["b"],
                       "fixed":[True]})
    mw.load_param_spreadsheet(spreadsheet=df)
    assert np.array_equal(mw.guesses,[5,3])
    assert np.array_equal(mw.fixed_mask,[False,True,False])
    assert np.array_equal(mw.bounds[0],[-10,-np.inf])
    assert np.array_equal(mw.bounds[1],[10,np.inf])
    assert np.array_equal(mw.priors[0],[-1,np.nan],equal_nan=True)
    assert np.array_equal(mw.priors[1],[1,np.nan],equal_nan=True)

    # ---------------------------------------------------------------
    # Send in parameter that is not part of the fit
    mw = ModelWrapper(model_to_test_wrap)
    df = pd.DataFrame({"param":["not_a_param"],
                       "fixed":[True]})
    with pytest.raises(ValueError):
        mw.load_param_spreadsheet(spreadsheet=df)

def test_ModelWrapper_model():

    def model_to_test_wrap(a=1,b=2,c=3,d="test",e=3): return a*b*c
    mw = ModelWrapper(model_to_test_wrap)
    m = mw.model
    
    assert hasattr(m,"__call__")
    assert m() == 1*2*3
    assert m([10,20,30]) == 10*20*30

    mw.a.guess = 20
    mw.a.fixed = True
    m = mw.model
    
    assert m() == 20*2*3
    assert m([20,30]) == 20*20*30
    with pytest.raises(ValueError):
        m([10,20,30]) # fixed parameter; can only take three args

def test_ModelWrapper_guesses():

    # test getter
    def model_to_test_wrap(a=1,b=2,c=3,d="test",e=3): return a*b*c
    mw = ModelWrapper(model_to_test_wrap)
    g = mw.guesses
    assert np.array_equal(g,[1,2,3])
    
    mw.c.guess = 20
    g = mw.guesses
    assert np.array_equal(g,[1,2,20])

    mw.b.fixed = True
    g = mw.guesses
    assert np.array_equal(g,[1,20])

    # test setter
    mw = ModelWrapper(model_to_test_wrap)
    assert np.array_equal(mw.guesses,[1,2,3])
    mw.guesses = [3,4,5]
    assert np.array_equal(mw.guesses,[3,4,5])

    # send in some stupid values
    with pytest.raises(ValueError):
        mw.guesses = 1

    with pytest.raises(ValueError):
        mw.guesses = [1,2,3,4]

    with pytest.raises(ValueError):
        mw.guesses = [1,2]

    with pytest.raises(ValueError):
        mw.guesses = ["a","b","c"]

    # out of bounds set
    mw.guesses = [0.5,0.5,0.5]
    mw.bounds = [[0,0,0],[1,1,1]]
    with pytest.raises(ValueError):
        mw.guesses = [10,10,10]


def test_ModelWrapper_bounds():

    def model_to_test_wrap(a=1,b=2,c=3,d="test",e=3): return a*b*c
    mw = ModelWrapper(model_to_test_wrap)
    b = mw.bounds
    assert np.array_equal(b.shape,[2,3])
    assert np.sum(np.isinf(b)) == 6
    assert b[0,1] == -np.inf
    assert b[1,1] == np.inf

    mw.b.bounds = (-10,10)
    b = mw.bounds
    assert np.array_equal(b.shape,[2,3])
    assert np.sum(np.isinf(b)) == 4
    assert b[0,1] == -10
    assert b[1,1] == 10

    mw.c.fixed = True
    b = mw.bounds
    assert np.array_equal(b.shape,[2,2])
    assert np.sum(np.isinf(b)) == 2
    assert b[0,1] == -10
    assert b[1,1] == 10

    # test setter
    mw = ModelWrapper(model_to_test_wrap)
    assert np.array_equal(mw.bounds.shape,[2,3])
    assert np.sum(np.isinf(mw.bounds)) == 6
    mw.bounds = [[-1,-1,-1],[100,100,100]]
    assert np.array_equal(mw.bounds[0,:],[-1,-1,-1])
    assert np.array_equal(mw.bounds[1,:],[100,100,100])

    # send in some stupid values
    with pytest.raises(ValueError):
        mw.bounds = 1

    with pytest.raises(ValueError):
        mw.bounds = np.ones((2,4),dtype=float)

    with pytest.raises(ValueError):
        mw.bounds = np.ones((3,2),dtype=float)

    with pytest.raises(ValueError):
        mw.bounds = [["a","b","c"],["e","f","g"]]

    with pytest.raises(ValueError):
        mw.bounds = np.nan*np.ones((2,3),dtype=float)

    # Same upper and lower bounds -- bad
    with pytest.raises(ValueError):
        mw.bounds = np.ones((2,3),dtype=float)

    # should work fine
    bounds = np.ones((2,3),dtype=float)
    bounds[0,:] = bounds[0,:]*-np.inf
    bounds[1,:] = bounds[1,:]*np.inf
    mw.bounds = bounds

def test_ModelWrapper_priors():

    # test gette

    def model_to_test_wrap(a=1,b=2,c=3,d="test",e=3): return a*b*c
    mw = ModelWrapper(model_to_test_wrap)
    p = mw.priors
    assert np.array_equal(p.shape,[2,3])
    assert np.sum(np.isnan(p)) == 6

    mw.a.prior = (10,1)
    p = mw.priors
    assert np.array_equal(p.shape,[2,3])
    assert np.sum(np.isnan(p)) == 4
    assert p[0,0] == 10
    assert p[1,0] == 1

    mw.b.fixed = True
    p = mw.priors
    assert np.array_equal(p.shape,[2,2])
    assert np.sum(np.isnan(p)) == 2
    assert p[0,0] == 10
    assert p[1,0] == 1

    # test setter
    mw = ModelWrapper(model_to_test_wrap)
    assert np.array_equal(mw.priors.shape,[2,3])
    assert np.sum(np.isnan(mw.priors)) == 6
    mw.priors = np.ones((2,3),dtype=float)
    assert np.sum(mw.priors) == 6

    # send in some stupid values
    with pytest.raises(ValueError):
        mw.priors = 1

    with pytest.raises(ValueError):
        mw.priors = np.ones((2,4),dtype=float)

    with pytest.raises(ValueError):
        mw.priors = np.ones((3,2),dtype=float)

    with pytest.raises(ValueError):
        mw.priors = [["a","b","c"],["e","f","g"]]

    with pytest.raises(ValueError):
        mw.priors = np.inf*np.ones((2,3),dtype=float)

    # should work fine
    mw.priors = np.nan*np.ones((2,3),dtype=float)

def test_ModelWrapper_names():

    # test getter

    def model_to_test_wrap(a=1,b=1,c="test",d=3): return a*b
    mw = ModelWrapper(model_to_test_wrap)
    n = mw.names
    assert len(n) == 2
    assert np.array_equal(n,["a","b"])
    
    mw.b.fixed = True
    n = mw.names
    assert len(n) == 1
    assert n[0] == "a"

    # test setter
    mw = ModelWrapper(model_to_test_wrap)
    assert np.array_equal(mw.names,["a","b"])
    mw.names = ["x","y"]
    assert np.array_equal(mw.names,["x","y"])

    # bad values
    with pytest.raises(ValueError):
        mw.names = 1

    with pytest.raises(ValueError):
        mw.names = [1,2,3]

    # this should work, but coerce
    mw.names = [1,2]
    assert np.array_equal(mw.names,["1","2"])


def test_ModelWrapper_values():

    # test getter

    def model_to_test_wrap(a=1,b=1,c="test",d=3): return a*b
    mw = ModelWrapper(model_to_test_wrap)
    v = mw.values
    assert len(v) == 2
    assert np.array_equal(v,[1,1])
    
    mw.a._value = 2 # cannot set value publicly, so hacking setter
    v = mw.values
    assert len(v) == 2
    assert np.array_equal(v,[2,1])
    
    mw.b.fixed = True
    v = mw.values
    assert len(v) == 1
    assert np.array_equal(v,[2])

    
def test_ModelWrapper_fixed_mask():

    # test getter

    def model_to_test_wrap(a=1,b=1,c="test",d=3): return a*b
    mw = ModelWrapper(model_to_test_wrap)
    fm = mw.fixed_mask
    assert len(fm) == 2
    assert np.array_equal(fm,[False,False])
    
    mw.a.fixed = True
    fm = mw.fixed_mask
    assert len(fm) == 2
    assert np.array_equal(fm,[True,False])

    # test setter
    def model_to_test_wrap(a=1,b=1,c="test",d=3): return a*b
    mw = ModelWrapper(model_to_test_wrap)
    assert np.array_equal(mw.fixed_mask,[False,False])
    mw.fixed_mask = [True,False]
    assert np.array_equal(mw.fixed_mask,[True,False])

    # some stupid values
    with pytest.raises(ValueError):
        mw.fixed_mask = 1

    with pytest.raises(ValueError):
        mw.fixed_mask = [True,False,False]

    with pytest.raises(ValueError):
        mw.fixed_mask = ["not",1]


def test_ModelWrapper_position_to_param():

    def model_to_test_wrap(a=1,b=1,c="test",d=3): return a*b
    mw = ModelWrapper(model_to_test_wrap)
    pp = mw.position_to_param
    assert pp is mw._position_to_param
    assert len(pp) == 2
    assert pp[0] == "a"
    assert pp[1] == "b"
    
    mw.a.fixed = True
    pp = mw.position_to_param
    assert len(pp) == 1
    assert pp[0] == "b"


def test_ModelWrapper_fit_parameters():
    
    def model_to_test_wrap(a=1,b=1,c="test",d=3): return a*b
    mw = ModelWrapper(model_to_test_wrap)
    fp = mw.fit_parameters
    assert fp is mw._mw_fit_parameters
    assert len(fp) == 2
    assert fp["a"].guess == 1
    assert fp["b"].guess == 1

def test_ModelWrapper_other_arguments():

    def model_to_test_wrap(a=1,b=1,c="test",d=3): return a*b
    mw = ModelWrapper(model_to_test_wrap)
    oa = mw.other_arguments
    assert oa is mw._mw_other_arguments
    assert len(oa) == 2
    assert oa["c"] == "test"
    assert oa["d"] == 3

# TESTS BELOW ARE OLD SET OF RELATIVELY INTEGRATED, NOT-QUITE-UNIT TESTS


def test_init(binding_curve_test_data):

    model_to_test_wrap = binding_curve_test_data["model_to_test_wrap"]

    # Make sure it correctly recognizes model parameters (takes first two b/c
    # can be coerced into float, not extra_stuff b/c not float, not K3 b/c
    # after non-fittable. )
    mw = ModelWrapper(model_to_test_wrap)
    assert isinstance(mw.K1,FitParameter)
    assert isinstance(mw.K2,FitParameter)
    assert not isinstance(mw.extra_stuff,FitParameter)
    assert not isinstance(mw.K3,FitParameter)

    assert mw.K1.guess == 0   # No guess specified --> should be 0.0
    assert mw.K2.guess == 20  # Default (guess) specified

    # Make sure that we only grab K1 if specified, not the other possible
    # parameters K2 and K3
    mw = ModelWrapper(model_to_test_wrap,fittable_params=["K1"])
    assert isinstance(mw.K1,FitParameter)
    assert not isinstance(mw.K2,FitParameter)
    assert not isinstance(mw.extra_stuff,FitParameter)

    # Make sure we can pass more than one fittable parameters
    mw = ModelWrapper(model_to_test_wrap,fittable_params=["K1","K2"])
    assert isinstance(mw.K1,FitParameter)
    assert isinstance(mw.K2,FitParameter)
    assert not isinstance(mw.extra_stuff,FitParameter)
    assert not isinstance(mw.K3,FitParameter)

    # Make sure we can grab a fittable parameter that would not normally
    # be used.
    mw = ModelWrapper(model_to_test_wrap,fittable_params=["K3","K2"])
    assert not isinstance(mw.K1,FitParameter)
    assert isinstance(mw.K2,FitParameter)
    assert not isinstance(mw.extra_stuff,FitParameter)
    assert isinstance(mw.K3,FitParameter)
    assert mw.K3.guess == 42

    # Recognizes bad manually passed parameter
    with pytest.raises(ValueError):
        mw = ModelWrapper(model_to_test_wrap,fittable_params=["not_real"])

    # Recognizes another type of bad manually passed parameter
    with pytest.raises(ValueError):
        mw = ModelWrapper(model_to_test_wrap,fittable_params=["extra_stuff"])

    # pass model that uses a reserved name as an argument
    def bad_model_with_reserved_name(guesses=2): return guesses
    with pytest.raises(ValueError):
        mw = ModelWrapper(bad_model_with_reserved_name)

    # pass model that uses a reserved name as a non-fittable argument
    def bad_model_with_reserved_name(a=2,b="test",guesses=2): return guesses
    with pytest.raises(ValueError):
        mw = ModelWrapper(bad_model_with_reserved_name)

    def model_with_only_kwargs(**kwargs): pass

    # error because no arguments
    with pytest.raises(ValueError):
        mw = ModelWrapper(model_with_only_kwargs)

    # This will work because we send in a kwarg
    mw = ModelWrapper(model_with_only_kwargs,
                      fittable_params=["a"])
    assert len(mw.fit_parameters) == 1
    assert mw.fit_parameters["a"].guess == 0

    def test_fcn(a=1,b=2,c="test",d=3,*args,**kwargs): pass
    mw = ModelWrapper(test_fcn)
    assert len(mw.fit_parameters) == 2
    assert mw.fit_parameters["a"].guess == 1
    assert mw.fit_parameters["b"].guess == 2

    mw = ModelWrapper(test_fcn,fittable_params=["e"])
    assert len(mw.fit_parameters) == 1
    assert mw.fit_parameters["e"].guess == 0

    # mix of function args and kwarg declared outside
    mw = ModelWrapper(test_fcn,fittable_params=["a","b","e"])
    assert len(mw.fit_parameters) == 3
    assert mw.fit_parameters["a"].guess == 1
    assert mw.fit_parameters["b"].guess == 2
    assert mw.fit_parameters["e"].guess == 0

    # include something not fittable
    with pytest.raises(ValueError):
        mw = ModelWrapper(test_fcn,fittable_params=["a","b","c","e"])


def test_expand_to_model_inputs(binding_curve_test_data):


    # Make sure function check works
    not_a_function = ["test",1,None,np.nan]
    for n in not_a_function:
        with pytest.raises(ValueError):
            ModelWrapper(n)

    model_to_test_wrap = binding_curve_test_data["model_to_test_wrap"]

    mw = ModelWrapper(model_to_test_wrap)

    # Make sure we get the right parameter names
    params = list(mw.fit_parameters.keys())
    assert params[0] == "K1" and params[1] == "K2"

    # Make sure we get the right non-fit-parameter names
    args = list(mw.other_arguments.keys())
    assert args[0] == "extra_stuff" and args[1] == "K3"

    # Check guesses
    assert np.array_equal(mw.guesses,np.array((0,20)))

    # Check bounds
    assert np.array_equal(mw.bounds[0],np.array((-np.inf,-np.inf)))
    assert np.array_equal(mw.bounds[1],np.array((np.inf,np.inf)))

    # Check names vector
    assert mw.names[0] == "K1"
    assert mw.names[1] == "K2"


def test_setting_guess(binding_curve_test_data):

    model_to_test_wrap = binding_curve_test_data["model_to_test_wrap"]

    # Default values set correctly
    mw = ModelWrapper(model_to_test_wrap)
    assert mw.K1.guess == 0
    assert mw.K2.guess == 20
    with pytest.raises(AttributeError):
        mw.K3.guess == 20

    # Setting K1 works but does not alter K2
    mw.K1 = 233
    assert mw.K1.guess == 233
    assert mw.K2.guess == 20

    # Setting K2.guess works
    mw.K2.guess = 32
    assert mw.K1.guess == 233
    assert mw.K2.guess == 32

    # Try, but fail, to set the guess with a string
    assert mw.K1.guess == 233
    with pytest.raises(ValueError):
        mw.K1.guess = "a string"
    assert mw.K1.guess == 233

    # Set guess with a string that can be coerced into a float
    assert mw.K1.guess == 233
    mw.K1.guess = "22"
    assert mw.K1.guess == 22

    # Test setting by fit_parameters dict
    assert mw.fit_parameters["K1"].guess == 22
    mw.fit_parameters["K1"].guess = 42
    assert mw.fit_parameters["K1"].guess == 42

def test_setting_bounds(binding_curve_test_data):

    model_to_test_wrap = binding_curve_test_data["model_to_test_wrap"]

    mw = ModelWrapper(model_to_test_wrap)

    # Set bounds
    assert np.array_equal(mw.bounds[0],np.array((-np.inf,-np.inf)))
    mw.K1.bounds = [0,500]
    assert np.array_equal(mw.K1.bounds,np.array([0,500]))

    # Try, but fail, to set bounds that are backwards
    with pytest.raises(ValueError):
        mw.K1.bounds = [500,-50]
    assert np.array_equal(mw.K1.bounds,np.array([0,500]))

    # Set bounds that do not encompass guess and make sure guess shifts
    mw.K1.guess = 0
    assert mw.K1.guess == 0
    with pytest.warns():
        mw.K1.bounds = [-500,-50]
    assert np.array_equal(mw.K1.bounds,np.array([-500,-50]))
    assert mw.K1.guess == -50

    # Test setting by fit_parameters dict
    mw = ModelWrapper(model_to_test_wrap)
    assert np.array_equal(mw.fit_parameters["K1"].bounds,np.array((-np.inf,np.inf)))
    mw.fit_parameters["K1"].bounds = [0,500]
    assert np.array_equal(mw.fit_parameters["K1"].bounds,np.array([0,500]))
    assert np.array_equal(mw.K1.bounds,np.array([0,500]))

def test_setting_name(binding_curve_test_data):

    model_to_test_wrap = binding_curve_test_data["model_to_test_wrap"]

    mw = ModelWrapper(model_to_test_wrap)

    # Test setting by mw.K
    assert mw.K1.name == "K1"
    mw.K1.name = "new_name"
    assert mw.K1.name == "new_name"
    assert mw.fit_parameters["K1"].name == "new_name"

    # Test setting via mw.fit_parameters
    assert mw.K2.name == "K2"
    assert mw.fit_parameters["K2"].name == "K2"
    mw.fit_parameters["K2"].name = "another name with spaces this time"
    assert mw.K2.name == "another name with spaces this time"
    assert mw.fit_parameters["K2"].name == "another name with spaces this time"

def test_setting_fixed(binding_curve_test_data):

    # This also tests the private function self._update_parameter_map b/c
    # changing fixed parameters is what changes the guesses and other properties

    model_to_test_wrap = binding_curve_test_data["model_to_test_wrap"]

    # Wrap model
    mw = ModelWrapper(model_to_test_wrap)
    assert mw.names[0] == "K1"
    assert mw.names[1] == "K2"

    # Fix one parameter
    mw.K1.fixed = True
    assert mw.names[0] == "K2"
    assert mw.guesses[0] == 20
    with pytest.raises(IndexError):
        mw.names[1]

    # Fix second parameter
    mw.K2.fixed = True
    with pytest.raises(IndexError):
        mw.names[0]

    # Unfix a parameter
    mw.K1.fixed = False
    assert mw.names[0] == "K1"
    assert mw.guesses[0] == 0
    with pytest.raises(IndexError):
        mw.names[1]

    # Try to fix a parameter that is not really a parameter
    with pytest.raises(AttributeError):
        mw.extra_stuff.fixed = True

    mw = ModelWrapper(model_to_test_wrap)
    assert mw.names[0] == "K1"
    assert mw.names[1] == "K2"


def test_setting_other_arguments(binding_curve_test_data):

    model_to_test_wrap = binding_curve_test_data["model_to_test_wrap"]

    mw = ModelWrapper(model_to_test_wrap)
    assert isinstance(mw.K1,FitParameter)
    assert isinstance(mw.K2,FitParameter)
    assert not isinstance(mw.extra_stuff,FitParameter)
    assert not isinstance(mw.K3,FitParameter)

    assert mw.other_arguments["extra_stuff"] == "test"
    assert mw.extra_stuff == "test"
    mw.other_arguments["extra_stuff"] = 19
    assert mw.extra_stuff == 19

def test_model_output(binding_curve_test_data):

    model_to_test_wrap = binding_curve_test_data["model_to_test_wrap"]

    mw = ModelWrapper(model_to_test_wrap)
    
    # Override K1 default to make nt all zero
    mw.K1.guess = 1

    # Test call with default parameters
    assert mw.model() == 1*20*42
    assert mw.model((1,20)) == 1*20*42
    assert mw.model((20,20)) == 20*20*42

    # Test pass through for bad argument
    with pytest.raises(ValueError):
        mw.model(("stupid",20))

    # test passing too many arguments
    with pytest.raises(ValueError):
        mw.model((20,20,42))

    # test passing too few arguments
    with pytest.raises(ValueError):
        mw.model((20,))

    # Test setting other argument that should change output
    mw.K3 = 14
    assert mw.model() == 1*20*14

