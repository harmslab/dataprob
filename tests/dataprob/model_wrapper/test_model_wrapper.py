import pytest

from dataprob.model_wrapper.model_wrapper import ModelWrapper
from dataprob.fit_param import FitParameter

import numpy as np
import pandas as pd

def test_ModelWrapper___init__():

    def model_to_test_wrap(a,b=2,c=3,d="test",e=3): return a*b*c

    # Test basic functionality
    mw = ModelWrapper(model_to_fit=model_to_test_wrap)
    assert mw._model_to_fit is model_to_test_wrap
    assert len(mw._mw_fit_parameters) == 3
    assert mw._mw_fit_parameters["a"].guess == 0
    assert mw._mw_fit_parameters["b"].guess == 2
    assert mw._mw_fit_parameters["c"].guess == 3
    
    assert len(mw._mw_other_arguments) == 2
    assert mw._mw_other_arguments["d"] == "test"
    assert mw._mw_other_arguments["e"] == 3

    # make sure fittable_params are being passed properly
    mw = ModelWrapper(model_to_test_wrap,fittable_params=["a"])
    assert mw._model_to_fit is model_to_test_wrap

    assert len(mw._mw_fit_parameters) == 1
    assert mw._mw_fit_parameters["a"].guess == 0
    
    assert len(mw._mw_other_arguments) == 4
    assert mw._mw_other_arguments["b"] == 2
    assert mw._mw_other_arguments["c"] == 3
    assert mw._mw_other_arguments["d"] == "test"
    assert mw._mw_other_arguments["e"] == 3



def test_ModelWrapper__mw_load_model():

    # Create a ModelWrapper that has just been initialized but has not
    # run load_model
    class TestModelWrapper(ModelWrapper):
        def __init__(self):
            self._mw_fit_parameters = {}
            self._mw_other_arguments = {}

    mw = TestModelWrapper()
    assert len(mw.__dict__) == 2

    def model_to_test_wrap(a,b=2,c=3,d="test",e=3): return a*b*c
    
    with pytest.raises(ValueError):
        mw._mw_load_model(model_to_fit="not_callable",
                          fittable_params=None)
    
    mw._mw_load_model(model_to_test_wrap,fittable_params=None)
    assert mw._model_to_fit is model_to_test_wrap

    # analyze_fcn_sig, reconcile_fittable, param_sanity check are all tested in
    # test_function_processing. We can basically only test results here. The 
    # model above covers almost the whole decision tree. Tests of complete 
    # decision tree follow. 

    assert len(mw._mw_fit_parameters) == 3
    assert mw._mw_fit_parameters["a"].guess == 0
    assert mw._mw_fit_parameters["b"].guess == 2
    assert mw._mw_fit_parameters["c"].guess == 3
    
    assert len(mw._mw_other_arguments) == 2
    assert mw._mw_other_arguments["d"] == "test"
    assert mw._mw_other_arguments["e"] == 3

    # This makes sure that the _update_parameter_map() call is happening. only 
    # test here because the logic of that call is tested in its own method call. 
    assert np.array_equal(mw._position_to_param,["a","b","c"])
    assert len(mw._mw_kwargs) == 5
    assert mw._mw_kwargs["a"] is None
    assert mw._mw_kwargs["b"] is None
    assert mw._mw_kwargs["c"] is None
    assert mw._mw_kwargs["d"] == "test"
    assert mw._mw_kwargs["e"] == 3

    # Now validate interaction with input function and fittable_params. Only
    # grab one argument. 
    mw = TestModelWrapper()
    assert len(mw.__dict__) == 2
    mw._mw_load_model(model_to_test_wrap,fittable_params=["a"])
    assert len(mw._mw_fit_parameters) == 1
    assert mw._mw_fit_parameters["a"].guess == 0
    
    assert len(mw._mw_other_arguments) == 4
    assert mw._mw_other_arguments["b"] == 2
    assert mw._mw_other_arguments["c"] == 3
    assert mw._mw_other_arguments["d"] == "test"
    assert mw._mw_other_arguments["e"] == 3

    # Now validate interaction with input function and fittable_params. Add 
    # argument that would not normally be grabbed. 
    mw = TestModelWrapper()
    assert len(mw.__dict__) == 2
    mw._mw_load_model(model_to_test_wrap,fittable_params=["a","e"])
    assert len(mw._mw_fit_parameters) == 2
    assert mw._mw_fit_parameters["a"].guess == 0
    assert mw._mw_fit_parameters["e"].guess == 3
    
    assert len(mw._mw_other_arguments) == 3
    assert mw._mw_other_arguments["b"] == 2
    assert mw._mw_other_arguments["c"] == 3
    assert mw._mw_other_arguments["d"] == "test"

    # Add argument not thought to be fittable by the parser.
    mw = TestModelWrapper()
    assert len(mw.__dict__) == 2
    with pytest.raises(ValueError):
        mw._mw_load_model(model_to_test_wrap,fittable_params=["a","d"])

    # fittable param that is not in arguments
    mw = TestModelWrapper()
    assert len(mw.__dict__) == 2
    with pytest.raises(ValueError):
        mw._mw_load_model(model_to_test_wrap,fittable_params=["w"])

    # not enough fittable params
    mw = TestModelWrapper()
    assert len(mw.__dict__) == 2
    with pytest.raises(ValueError):
        mw._mw_load_model(model_to_test_wrap,fittable_params=[])
    
    # send in a model that is only kwargs and make sure it still gets a fittable
    # param.
    def model_to_test_wrap(**kwargs): return kwargs["a"]
    mw = TestModelWrapper()
    with pytest.raises(ValueError):
        mw._mw_load_model(model_to_test_wrap,fittable_params=None)
        
    mw = TestModelWrapper()
    mw._mw_load_model(model_to_test_wrap,fittable_params=["a"])
    assert len(mw._mw_fit_parameters) == 1
    assert mw._mw_fit_parameters["a"].guess == 0
    assert len(mw._mw_other_arguments) == 0


def test_ModelWrapper__setattr__():

    def model_to_test_wrap(a=1,b=2,c=3,d="test",e=3): return a*b*c
    mw = ModelWrapper(model_to_test_wrap)
    
    # test setting fit parameter
    assert mw._mw_fit_parameters["a"].guess == 1
    mw.__setattr__("a",10)
    assert mw._mw_fit_parameters["a"].guess == 10

    # test setting other parameter
    assert mw._mw_other_arguments["d"] == "test"
    mw.__setattr__("d", 4)
    assert mw._mw_other_arguments["d"] == 4

    # test setting __dict__ parameter
    assert "something_else" not in mw.__dict__
    mw.__setattr__("something_else",10)
    assert mw.__dict__["something_else"] == 10


def test_ModelWrapper___getattr__():

    def model_to_test_wrap(a=1,b=2,c=3,d="test",e=3): return a*b*c
    mw = ModelWrapper(model_to_test_wrap)
    mw.blah = "non_fit_attribute"

    # test getting on fit and other  parameters
    assert mw.__getattr__("a") is mw._mw_fit_parameters["a"]
    assert mw.__getattr__("e") is mw._mw_other_arguments["e"]

    # test __dict__ get
    assert mw.__getattr__("blah") == "non_fit_attribute"

    # test __getattribute__ fallback got @property getter
    assert mw.__getattr__("fit_parameters") is mw._mw_fit_parameters
    
    # test __getattribute__ fallback for built in method
    assert mw.__getattr__("__init__") == mw.__init__

    # Final fail
    with pytest.raises(AttributeError):
        mw.__getattr__("not_an_attribute")

    

def test_ModelWrapper__update_parameter_map():

    def model_to_test_wrap(a=1,b=2,c=3,d="test",e=3): return a*b*c
    mw = ModelWrapper(model_to_test_wrap)
    
    # Check initial configuration after __init__
    assert np.array_equal(mw._position_to_param,["a","b","c"])
    assert len(mw._mw_other_arguments) == 2
    assert mw._mw_other_arguments["d"] == "test"
    assert mw._mw_other_arguments["e"] == 3
    assert len(mw._mw_kwargs) == 5
    assert mw._mw_kwargs["a"] is None
    assert mw._mw_kwargs["b"] is None
    assert mw._mw_kwargs["c"] is None
    assert mw._mw_kwargs["d"] == "test"
    assert mw._mw_kwargs["e"] == 3

    # Edit parameters
    mw.a.guess = 10
    mw.a.fixed = True

    # Make sure no change
    assert np.array_equal(mw._position_to_param,["a","b","c"])
    assert len(mw._mw_other_arguments) == 2
    assert mw._mw_other_arguments["d"] == "test"
    assert mw._mw_other_arguments["e"] == 3
    assert len(mw._mw_kwargs) == 5
    assert mw._mw_kwargs["a"] is None
    assert mw._mw_kwargs["b"] is None
    assert mw._mw_kwargs["c"] is None
    assert mw._mw_kwargs["d"] == "test"
    assert mw._mw_kwargs["e"] == 3

    # Run function
    mw._update_parameter_map()

    # Check for expected output
    assert np.array_equal(mw._position_to_param,["b","c"])
    assert len(mw._mw_other_arguments) == 2
    assert mw._mw_other_arguments["d"] == "test"
    assert mw._mw_other_arguments["e"] == 3
    assert len(mw._mw_kwargs) == 5
    assert mw._mw_kwargs["a"] == 10
    assert mw._mw_kwargs["b"] is None
    assert mw._mw_kwargs["c"] is None
    assert mw._mw_kwargs["d"] == "test"
    assert mw._mw_kwargs["e"] == 3
    


def test_ModelWrapper__mw_observable():

    # Simple model
    def model_to_test_wrap(a=1,b=2,c=3,d="test",e=3): return a*b*c
    mw = ModelWrapper(model_to_test_wrap)
    
    # internal parameters
    assert mw._mw_observable() == 1*2*3
    
    # bad parameters -- wrong length
    with pytest.raises(ValueError):
        mw._mw_observable([1,2])
    
    with pytest.raises(ValueError):
        mw._mw_observable([1,2,3,4])

    # check valid pass of parameter
    assert mw._mw_observable([3,4,5]) == 3*4*5

    # fix parameter
    mw.b.fixed = True
    mw._update_parameter_map()
    assert mw._mw_observable([3,4]) == 3*2*4 #(a*fixed(b)*c)

    # now fail because too many params
    with pytest.raises(ValueError):
        mw._mw_observable([3,4,5])

    def model_to_test_wrap(a=1,b=2,c=3,d="test",e=3): raise ValueError
    mw = ModelWrapper(model_to_test_wrap)
    with pytest.raises(RuntimeError):
        mw._mw_observable()



def test_ModelWrapper_load_fit_result(binding_curve_test_data,fitter_object):

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

