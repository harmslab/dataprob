"""
Relatively integrated, not-quite-unit-level tests of the ModelWrapper and its 
interaction with the FitParameter and Fitter classes.
"""

import pytest

from dataprob.model_wrapper.model_wrapper import ModelWrapper
from dataprob.fit_param import FitParameter

import numpy as np

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
    with pytest.raises(RuntimeError):
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

