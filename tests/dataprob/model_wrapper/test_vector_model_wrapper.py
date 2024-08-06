import pytest

from dataprob.model_wrapper.vector_model_wrapper import VectorModelWrapper

import numpy as np

def test_VectorModelWrapper__init__():

    # Check basic wrapping
    def test_fcn(some_array,a,b="test"): return some_array[0] + some_array[1] + some_array[2]

    mw = VectorModelWrapper(model_to_fit=test_fcn,
                            fittable_params={"x":1,"y":2,"z":3})
    
    assert len(mw.fit_parameters) == 3
    assert mw.fit_parameters["x"].guess == 1
    assert mw.fit_parameters["y"].guess == 2
    assert mw.fit_parameters["z"].guess == 3
    assert mw.other_arguments["b"] == "test"
    assert mw._mw_observable() == 6

def test_VectorModelWrapper__mw_load_model():

    # not callable
    with pytest.raises(ValueError):
        mw = VectorModelWrapper(model_to_fit="not_callable",
                                fittable_params={"x":1,"y":2,"z":3})
    
    # No args
    def test_fcn(): pass
    with pytest.raises(ValueError):
        mw = VectorModelWrapper(model_to_fit=test_fcn,
                                fittable_params={"x":1,"y":2,"z":3})
        
    # no fittable_params
    def test_fcn(x): pass
    with pytest.raises(ValueError):
        mw = VectorModelWrapper(model_to_fit=test_fcn,
                                fittable_params={}) 
        
    # bad fittable_params
    def test_fcn(x): pass
    with pytest.raises(ValueError):
        mw = VectorModelWrapper(model_to_fit=test_fcn,
                                fittable_params=None) 
    
    # fittable_param dict, bad value
    def test_fcn(x): pass
    with pytest.raises(ValueError):
        mw = VectorModelWrapper(model_to_fit=test_fcn,
                                fittable_params={"a":"test"}) 
        
    # fittable_param dict, bad value because it matches secondary 
    # argument to function
    def test_fcn(x,a): pass
    with pytest.raises(ValueError):
        mw = VectorModelWrapper(model_to_fit=test_fcn,
                                fittable_params={"a":1.0}) 
    
    # extra argument conflicts with attribute of the class
    def test_fcn(x,guesses): pass
    with pytest.raises(ValueError):
        mw = VectorModelWrapper(model_to_fit=test_fcn,
                                fittable_params={"a":1.0}) 
    
    # fittable_param dict, good value
    def test_fcn(x): pass
    mw = VectorModelWrapper(model_to_fit=test_fcn,
                            fittable_params={"a":20}) 
    assert mw._model_to_fit is test_fcn
    assert mw.fit_parameters["a"].value == 20

    # fittable_param list
    def test_fcn(x): pass
    mw = VectorModelWrapper(model_to_fit=test_fcn,
                            fittable_params=["a","b"]) 
    assert mw._model_to_fit is test_fcn
    assert mw.fit_parameters["a"].value == 0
    assert mw.fit_parameters["b"].value == 0

def test__update_parameter_map():

    # fittable_param dict, good value
    def test_fcn(x,z="test"): pass
    mw = VectorModelWrapper(model_to_fit=test_fcn,
                            fittable_params={"a":20,"b":30}) 
    
    # Check initial sets
    assert mw._model_to_fit is test_fcn
    assert mw.fit_parameters["a"].value == 20
    assert mw.fit_parameters["b"].value == 30
    assert mw.other_arguments["z"] == "test"

    # _update_param_map ran implicitly. Make sure it's set correctly. 
    assert mw._param_vector_length == 2
    assert mw._num_unfixed == 2
    assert np.array_equal(mw._unfixed_param_mask,[True,True])
    assert np.array_equal(mw._current_parameter_values,[20,30])

    # Fix a value
    mw.a.fixed = True

    # Nothing should have propagated yet because we have not run update
    assert mw._param_vector_length == 2
    assert mw._num_unfixed == 2
    assert np.array_equal(mw._unfixed_param_mask,[True,True])
    assert np.array_equal(mw._current_parameter_values,[20,30])

    # update
    mw._update_parameter_map()

    # Now should have changed
    assert mw._param_vector_length == 2
    assert mw._num_unfixed == 1
    assert np.array_equal(mw._unfixed_param_mask,[False,True])
    assert np.array_equal(mw._current_parameter_values,[20,30])

    # Alter who is fixed and their values
    mw.a.fixed = False
    mw.a.guess = 30

    mw.b.fixed = True
    mw.b.guess = 100
    
    # No change yet
    assert mw._param_vector_length == 2
    assert mw._num_unfixed == 1
    assert np.array_equal(mw._unfixed_param_mask,[False,True])
    assert np.array_equal(mw._current_parameter_values,[20,30])

    # Run function
    mw._update_parameter_map()

    # Should change
    assert mw._param_vector_length == 2
    assert mw._num_unfixed == 1
    assert np.array_equal(mw._unfixed_param_mask,[True,False])
    assert np.array_equal(mw._current_parameter_values,[30,100])

    # Fix both and run
    mw.a.fixed = True
    mw.b.fixed = True
    mw._update_parameter_map()

    assert mw._param_vector_length == 2
    assert mw._num_unfixed == 0
    assert np.array_equal(mw._unfixed_param_mask,[False,False])
    assert np.array_equal(mw._current_parameter_values,[30,100])
    

def test_VectorModelWrapper__mw_observable():

    # fittable_param dict, good value
    def test_fcn(x,z="test"): return x[0] + x[1] + x[2]
    mw = VectorModelWrapper(model_to_fit=test_fcn,
                            fittable_params={"a":20,"b":30,"c":50}) 
    
    # Check initial sets
    assert mw._model_to_fit is test_fcn
    assert mw.fit_parameters["a"].value == 20
    assert mw.fit_parameters["b"].value == 30
    assert mw.fit_parameters["c"].value == 50
    assert mw.other_arguments["z"] == "test"

    # basic check. Does it run with parameters sent in?
    result = mw._mw_observable([1,2,3])
    assert result == 6

    # basic check. no parameters sent in -- pulled from the parameter guessess
    result = mw._mw_observable(params=None)
    assert result == 20 + 30 + 50

    # fix a parameter. should still work
    mw.a.fixed = True
    mw._update_parameter_map()
    result = mw._mw_observable(params=None)
    assert result == 20 + 30 + 50

    mw.a.fixed = True
    mw.b.fixed = True
    mw._update_parameter_map()
    result = mw._mw_observable(params=[1,2,3])
    assert result == 6

    # Should give fixed values for a and b plus what we sent in for c
    result = mw._mw_observable(params=[1000])
    assert result == 20 + 30 + 1000

    # wrong number of parameters
    with pytest.raises(ValueError):
        result = mw._mw_observable(params=[1,2])

    # wrong number of parameters
    with pytest.raises(ValueError):
        result = mw._mw_observable(params=[1,2,3,4])

    # wrong number of parameters
    with pytest.raises(ValueError):
        result = mw._mw_observable(params=[])

    # Test error catching from model
    def test_fcn(x,z="test"): raise TypeError
    mw = VectorModelWrapper(model_to_fit=test_fcn,
                            fittable_params={"a":20,"b":30,"c":50}) 
    with pytest.raises(RuntimeError):
        mw._mw_observable()


def test_VectorModelWrapper_model():

    # fittable_param dict, good value
    def test_fcn(x,z="test"): return x[0] + x[1] + x[2]
    mw = VectorModelWrapper(model_to_fit=test_fcn,
                            fittable_params={"a":20,"b":30,"c":50}) 
    
    assert mw.model() == 20 + 30  + 50
    assert mw.model([1,2,3]) == 1 + 2 + 3
    
    # fix parameter -- should propagate properly
    mw.a.fixed = True
    assert mw.model([2,3]) == 20 + 2 + 3
