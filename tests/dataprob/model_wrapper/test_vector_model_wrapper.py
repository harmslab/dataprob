import pytest

from dataprob.model_wrapper.vector_model_wrapper import VectorModelWrapper

import numpy as np
import pandas as pd

def test_VectorModelWrapper__init__():

    # Check basic wrapping
    def test_fcn(some_array,a,b="test"): return some_array[0] + some_array[1] + some_array[2]

    mw = VectorModelWrapper(model_to_fit=test_fcn,
                            fit_parameters={"x":1,"y":2,"z":3})
    
    assert len(mw.param_df) == 3
    assert mw.param_df.loc["x","guess"] == 1
    assert mw.param_df.loc["y","guess"] == 2
    assert mw.param_df.loc["z","guess"] == 3
    assert mw.non_fit_kwargs["a"] is None
    assert mw.non_fit_kwargs["b"] == "test"
    assert mw.model() == 6

def test_VectorModelWrapper__load_model():

    # Create a ModelWrapper that has just been initialized but has not
    # run load_model
    class TestVectorModelWrapper(VectorModelWrapper):
        def __init__(self):
            self._default_guess = 0.0
            self._param_df = pd.DataFrame({"name":[]})
            self._non_fit_kwargs = {}
            

    mw = TestVectorModelWrapper()
    
    # No args
    def test_fcn(): pass
    with pytest.raises(ValueError):
        mw._load_model(model_to_fit=test_fcn,
                       fit_parameters={"x":1,"y":2,"z":3},
                       non_fit_kwargs=None)
        
    # no fit_parameters
    def test_fcn(x): pass
    with pytest.raises(ValueError):
        mw._load_model(model_to_fit=test_fcn,
                       fit_parameters={},
                       non_fit_kwargs=None) 
        
    # bad fit_parameters
    def test_fcn(x): pass
    with pytest.raises(ValueError):
        mw._load_model(model_to_fit=test_fcn,
                       fit_parameters=None,
                       non_fit_kwargs=None) 
    
    # bad fit_parameters --> has same name as first arg
    def test_fcn(x): pass
    with pytest.raises(ValueError):
        mw._load_model(model_to_fit=test_fcn,
                       fit_parameters=["x"],
                       non_fit_kwargs=None) 
    

    # fittable_param dict, bad value
    def test_fcn(x): pass
    with pytest.raises(ValueError):
        mw._load_model(model_to_fit=test_fcn,
                       fit_parameters={"a":"test"},
                       non_fit_kwargs=None) 
        
    # fittable_param dict, bad value because it matches secondary 
    # argument to function
    def test_fcn(x,a): pass
    with pytest.raises(ValueError):
        mw._load_model(model_to_fit=test_fcn,
                       fit_parameters={"a":1.0},
                       non_fit_kwargs=None) 
    
    
    # fittable_param dict, good value
    mw = TestVectorModelWrapper()
    def test_fcn(x): pass
    mw._load_model(model_to_fit=test_fcn,
                   fit_parameters={"a":20},
                   non_fit_kwargs=None) 
    assert mw._model_to_fit is test_fcn
    assert mw.param_df.loc["a","guess"] == 20

    # fittable_param list
    mw = TestVectorModelWrapper()
    def test_fcn(x): pass
    mw._load_model(model_to_fit=test_fcn,
                   fit_parameters=["a","b"],
                   non_fit_kwargs=None) 
    assert mw._model_to_fit is test_fcn
    assert mw.param_df.loc["a","guess"] == 0
    assert mw.param_df.loc["b","guess"] == 0
    assert np.array_equal(mw._fit_params_in_order,["a","b"])

    # fittable_param dict, good value, extra args in function
    mw = TestVectorModelWrapper()
    def test_fcn(x,b,c=6): pass
    mw._load_model(model_to_fit=test_fcn,
                   fit_parameters={"a":20},
                   non_fit_kwargs=None) 
    assert mw._model_to_fit is test_fcn
    assert mw.param_df.loc["a","guess"] == 20
    assert len(mw._non_fit_kwargs) == 2
    assert mw._non_fit_kwargs["b"] is None
    assert mw._non_fit_kwargs["c"] == 6

    # fittable_param dict, good value, extra args in function, with kwargs. 
    # kwargs ignored because no new non_fit_kwargs
    mw = TestVectorModelWrapper()
    def test_fcn(x,b,c=6,**kwargs): pass
    mw._load_model(model_to_fit=test_fcn,
                   fit_parameters={"a":20},
                   non_fit_kwargs=None) 
    assert mw._model_to_fit is test_fcn
    assert mw.param_df.loc["a","guess"] == 20
    assert len(mw._non_fit_kwargs) == 2
    assert mw._non_fit_kwargs["b"] is None
    assert mw._non_fit_kwargs["c"] == 6

    # fittable_param dict, good value, extra args in function, with kwargs. 
    # new non_fit_kwargs in kwarg
    mw = TestVectorModelWrapper()
    def test_fcn(x,b,c=6,**kwargs): pass
    mw._load_model(model_to_fit=test_fcn,
                   fit_parameters={"a":20},
                   non_fit_kwargs={"c":16,"d":17}) 
    assert mw._model_to_fit is test_fcn
    assert mw.param_df.loc["a","guess"] == 20
    assert len(mw._non_fit_kwargs) == 3
    assert mw._non_fit_kwargs["b"] is None
    assert mw._non_fit_kwargs["c"] == 16
    assert mw._non_fit_kwargs["d"] == 17

    # Should not work if first arg is in non_fit_kwargs
    mw = TestVectorModelWrapper()
    def test_fcn(x,b,c=6,**kwargs): pass
    with pytest.raises(ValueError):
        mw._load_model(model_to_fit=test_fcn,
                    fit_parameters={"a":20},
                    non_fit_kwargs={"x":1})

    # Should not work if fittable and non_fit_kwargs have same args
    mw = TestVectorModelWrapper()
    def test_fcn(x,b,c=6,**kwargs): pass
    with pytest.raises(ValueError):
        mw._load_model(model_to_fit=test_fcn,
                       fit_parameters=["a"],
                       non_fit_kwargs={"a":20})
    

    # Should not work if non_fit_kwargs is not in arg list and no kwargs
    mw = TestVectorModelWrapper()
    def test_fcn(x,b,c=6): pass
    with pytest.raises(ValueError):
        mw._load_model(model_to_fit=test_fcn,
                       fit_parameters={"a":20},
                       non_fit_kwargs={"d":"missing"}), 


    
def test_VectorModelWrapper__finalize_params():

    def model_to_test_wrap(a,b,c=3): return a[0]*a[1]*b*c
    mw = VectorModelWrapper(model_to_test_wrap,
                            fit_parameters={"x":1,"y":2})
    
    # Check initial configuration after __init__
    assert np.array_equal(mw._fit_params_in_order,["x","y"])
    assert mw._param_df.loc["x","guess"] == 1
    assert np.array_equal(mw._unfixed_mask,[True,True])
    assert len(mw._non_fit_kwargs) == 2
    assert mw._non_fit_kwargs["b"] is None
    assert mw._non_fit_kwargs["c"] == 3
    
    # Edit parameters
    mw.param_df.loc["x","guess"] = 10
    mw.param_df.loc["x","fixed"] = True

    assert np.array_equal(mw._fit_params_in_order,["x","y"])
    assert mw._param_df.loc["x","guess"] == 10
    assert np.array_equal(mw._unfixed_mask,[True,True])
    assert len(mw._non_fit_kwargs) == 2
    assert mw._non_fit_kwargs["b"] is None
    assert mw._non_fit_kwargs["c"] == 3

    # Run function
    mw.finalize_params()

    # Check for expected output
    assert np.array_equal(mw._fit_params_in_order,["x","y"])
    assert mw._param_df.loc["x","guess"] == 10
    assert np.array_equal(mw._unfixed_mask,[False,True])
    assert len(mw._non_fit_kwargs) == 2
    assert mw._non_fit_kwargs["b"] is None
    assert mw._non_fit_kwargs["c"] == 3

    # send in bad edit -- finalize should catch it
    mw.param_df.loc["a","fixed"] = True
    np.array_equal(mw.param_df.index,["x","y","a"])
    with pytest.raises(ValueError):
        mw.finalize_params()
    
def test_VectorModelWrapper_model():

    # fittable_param dict, good value
    def test_fcn(x,z="test"): return x[0] + x[1] + x[2]
    mw = VectorModelWrapper(model_to_fit=test_fcn,
                            fit_parameters={"a":20,"b":30,"c":50}) 
    
    # Check initial sets
    assert mw._model_to_fit is test_fcn
    assert mw.param_df.loc["a","guess"] == 20
    assert mw.param_df.loc["b","guess"] == 30
    assert mw.param_df.loc["c","guess"] == 50
    assert mw.non_fit_kwargs["z"] == "test"

    # not enough
    with pytest.raises(ValueError):
        mw.model(params=[1])

    # too many parameters
    with pytest.raises(ValueError):
        mw.model(params=[1,2,3,4,5,6,7])

     # something that is not even an array of numbers
    with pytest.raises(ValueError):
        result = mw.model(params="stupid")

    print("XNAY",mw.param_df)

    # basic check. Does it run with parameters sent in?
    result = mw.model([1,2,3])
    assert result == 6

    print("HERE",mw.param_df)

    # basic check. no parameters sent in -- pulled from the parameter guessess
    result = mw.model(params=None)
    assert result == 20 + 30 + 50

    # fix a parameter. should still work
    mw.param_df.loc["a","fixed"] = True
    result = mw.model(params=None)
    assert result == 20 + 30 + 50

    # Guesses override what we sent in 
    mw.param_df.loc["a","fixed"] = True
    mw.param_df.loc["b","fixed"] = True
    result = mw.model(params=[1,2,3])
    assert result == 20 + 30 + 3

    # Should give fixed values for a and b plus what we sent in for c
    result = mw.model(params=[1000])
    assert result == 20 + 30 + 1000

    # change "a"
    mw.param_df.loc["a","guess"] = 10

    # make sure it recognizes fix and guess
    assert mw.model([1000]) == 10 + 30 + 1000
    assert mw.model([1,2,1000]) == 10 + 30 + 1000

    # Test error catching from model
    def test_fcn(x,z="test"): raise TypeError
    mw = VectorModelWrapper(model_to_fit=test_fcn,
                            fit_parameters={"a":20,"b":30,"c":50}) 
    with pytest.raises(RuntimeError):
        mw.model()

    
def test_VectorModelWrapper_fast_model():

    # fittable_param dict, good value
    def test_fcn(x,z="test"): return x[0] + x[1] + x[2]
    mw = VectorModelWrapper(model_to_fit=test_fcn,
                            fit_parameters={"a":20,"b":30,"c":50}) 
    
    # Make sure it is callable and takes arguments
    assert mw.fast_model([10,20,30]) == 10 + 20 + 30

    # Make sure it calls finalize. 
    # Run model and _mw_observable
    assert mw.model([1,2,3]) == 1 + 2 + 3
    assert mw.fast_model(np.array([1,2,3])) == 1 + 2 + 3

    # fix and change "a"
    mw.param_df.loc["a","fixed"] = True
    mw.param_df.loc["a","guess"] = 10

    # fast_model is not aware of this 
    assert mw.fast_model(np.array([1,2,3])) == 1 + 2 + 3
    
    mw.finalize_params()

    # and now fast_model should be too if finalized
    assert mw.fast_model(np.array([2,3])) == 10 + 2 + 3
