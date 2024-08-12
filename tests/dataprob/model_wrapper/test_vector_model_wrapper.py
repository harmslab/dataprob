import pytest

from dataprob.model_wrapper.vector_model_wrapper import VectorModelWrapper

import numpy as np
import pandas as pd

def xtest_VectorModelWrapper__init__():

    # Check basic wrapping
    def test_fcn(some_array,a,b="test"): return some_array[0] + some_array[1] + some_array[2]

    mw = VectorModelWrapper(model_to_fit=test_fcn,
                            fittable_params={"x":1,"y":2,"z":3})
    
    assert len(mw.param_df) == 3
    assert mw.param_df.loc["x","guess"] == 1
    assert mw.param_df.loc["y","guess"] == 2
    assert mw.param_df.loc["z","guess"] == 3
    assert mw.other_arguments["a"] == 0
    assert mw.other_arguments["b"] == "test"
    assert mw._mw_observable() == 6

def test_VectorModelWrapper__load_model():

    # Create a ModelWrapper that has just been initialized but has not
    # run load_model
    class TestVectorModelWrapper(VectorModelWrapper):
        def __init__(self):
            self._param_df = pd.DataFrame({"name":[]})
            self._other_arguments = {}
            self._default_guess = 0.0

    mw = TestVectorModelWrapper()

    # not callable
    with pytest.raises(ValueError):
        mw._load_model(model_to_fit="not_callable",
                       fittable_params={"x":1,"y":2,"z":3})
    
    # No args
    def test_fcn(): pass
    with pytest.raises(ValueError):
        mw._load_model(model_to_fit=test_fcn,
                       fittable_params={"x":1,"y":2,"z":3})
        
    # no fittable_params
    def test_fcn(x): pass
    with pytest.raises(ValueError):
        mw._load_model(model_to_fit=test_fcn,
                       fittable_params={}) 
        
    # bad fittable_params
    def test_fcn(x): pass
    with pytest.raises(ValueError):
        mw._load_model(model_to_fit=test_fcn,
                       fittable_params=None) 
    
    # fittable_param dict, bad value
    def test_fcn(x): pass
    with pytest.raises(ValueError):
        mw._load_model(model_to_fit=test_fcn,
                       fittable_params={"a":"test"}) 
        
    # fittable_param dict, bad value because it matches secondary 
    # argument to function
    def test_fcn(x,a): pass
    with pytest.raises(ValueError):
        mw._load_model(model_to_fit=test_fcn,
                       fittable_params={"a":1.0}) 
    
    # extra argument conflicts with attribute of the class
    def test_fcn(x,model): pass
    with pytest.raises(ValueError):
        mw._load_model(model_to_fit=test_fcn,
                       fittable_params={"a":1.0}) 
    
    # fittable_param dict, good value
    mw = TestVectorModelWrapper()
    def test_fcn(x): pass
    mw._load_model(model_to_fit=test_fcn,
                   fittable_params={"a":20}) 
    assert mw._model_to_fit is test_fcn
    assert mw.param_df.loc["a","guess"] == 20

    # fittable_param list
    mw = TestVectorModelWrapper()
    def test_fcn(x): pass
    mw._load_model(model_to_fit=test_fcn,
                   fittable_params=["a","b"]) 
    assert mw._model_to_fit is test_fcn
    assert mw.param_df.loc["a","guess"] == 0
    assert mw.param_df.loc["b","guess"] == 0
    assert np.array_equal(mw._fit_params_in_order,["a","b"])

    # fittable_param dict, good value, extra args in function
    mw = TestVectorModelWrapper()
    def test_fcn(x,b,c=6): pass
    mw._load_model(model_to_fit=test_fcn,
                   fittable_params={"a":20}) 
    assert mw._model_to_fit is test_fcn
    assert mw.param_df.loc["a","guess"] == 20
    assert len(mw._other_arguments) == 2
    assert mw._other_arguments["b"] is None
    assert mw._other_arguments["c"] == 6

    
def test_ModelWrapper__finalize_params():

    def model_to_test_wrap(a,b,c=3): return a[0]*a[1]*b*c
    mw = VectorModelWrapper(model_to_test_wrap,
                            fittable_params={"x":1,"y":2})
    
    # Check initial configuration after __init__
    assert np.array_equal(mw._fit_params_in_order,["x","y"])
    assert mw._param_df.loc["x","guess"] == 1
    assert np.array_equal(mw._unfixed_mask,[True,True])
    assert len(mw._other_arguments) == 2
    assert mw._other_arguments["b"] is None
    assert mw._other_arguments["c"] == 3
    
    # Edit parameters
    mw.x = 10
    mw.param_df.loc["x","fixed"] = True

    assert np.array_equal(mw._fit_params_in_order,["x","y"])
    assert mw._param_df.loc["x","guess"] == 10
    assert np.array_equal(mw._unfixed_mask,[True,True])
    assert len(mw._other_arguments) == 2
    assert mw._other_arguments["b"] is None
    assert mw._other_arguments["c"] == 3

    # Run function
    mw.finalize_params()

    # Check for expected output
    assert np.array_equal(mw._fit_params_in_order,["x","y"])
    assert mw._param_df.loc["x","guess"] == 10
    assert np.array_equal(mw._unfixed_mask,[False,True])
    assert len(mw._other_arguments) == 2
    assert mw._other_arguments["b"] is None
    assert mw._other_arguments["c"] == 3

    # send in bad edit -- finalize should catch it
    mw.param_df.loc["a","fixed"] = True
    np.array_equal(mw.param_df.index,["x","y","a"])
    with pytest.raises(ValueError):
        mw.finalize_params()
    
def test_VectorModelWrapper__mw_observable():

    # fittable_param dict, good value
    def test_fcn(x,z="test"): return x[0] + x[1] + x[2]
    mw = VectorModelWrapper(model_to_fit=test_fcn,
                            fittable_params={"a":20,"b":30,"c":50}) 
    
    # Check initial sets
    assert mw._model_to_fit is test_fcn
    assert mw.param_df.loc["a","guess"] == 20
    assert mw.param_df.loc["b","guess"] == 30
    assert mw.param_df.loc["c","guess"] == 50
    assert mw.other_arguments["z"] == "test"

    # not enough
    with pytest.raises(ValueError):
        mw._mw_observable(params=[1])

    # too many parameters
    with pytest.raises(ValueError):
        mw._mw_observable(params=[1,2,3,4,5,6,7])


    # basic check. Does it run with parameters sent in?
    result = mw._mw_observable([1,2,3])
    assert result == 6

    # basic check. no parameters sent in -- pulled from the parameter guessess
    result = mw._mw_observable(params=None)
    assert result == 20 + 30 + 50

    # fix a parameter. should still work
    mw.param_df.loc["a","fixed"] = True
    mw.finalize_params()
    result = mw._mw_observable(params=None)
    assert result == 20 + 30 + 50

    mw.param_df.loc["a","fixed"] = True
    mw.param_df.loc["b","fixed"] = True
    mw.finalize_params()
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

    # light wrapper for _mw_observable (tested elsewhere). Make sure finalize
    # runs and that it works as advertised but do not test deeply

    # fittable_param dict, good value
    def test_fcn(x,z="test"): return x[0] * x[1] * x[2]
    mw = VectorModelWrapper(model_to_fit=test_fcn,
                            fittable_params={"a":20,"b":30,"c":50}) 
    
    m = mw.model

    # Make sure it is callable and takes arguments
    assert hasattr(m,"__call__")
    assert m() == 20*30*50
    assert m([10,20,30]) == 10*20*30

    # Make sure it calls finalize. 
    # Run model and _mw_observable
    assert mw.model([1,2,3]) == 1*2*3
    assert mw._mw_observable([1,2,3]) == 1*2*3

    # fix and change "a"
    mw.param_df.loc["a","fixed"] = True
    mw.param_df.loc["a","guess"] = 10

    # mw_observable is not aware of this 
    assert mw._mw_observable([1,2,3]) == 1*2*3
    with pytest.raises(ValueError):
        mw._mw_observable([2,3])

    # but model is because it calls finalize
    assert mw.model([2,3]) == 10*2*3

    # and now mw_observable should be too
    assert mw._mw_observable([2,3]) == 10*2*3