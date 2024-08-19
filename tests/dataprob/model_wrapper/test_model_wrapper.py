import pytest

from dataprob.model_wrapper.model_wrapper import ModelWrapper

import numpy as np
import pandas as pd

def test_ModelWrapper___init__():

    def model_to_test_wrap(a,b=2,c=3,d="test",e=3): return a*b*c

    # Test argument checking
    with pytest.raises(ValueError):
        mw = ModelWrapper(model_to_fit="not_callable")
    
    # Bad fittable_params
    with pytest.raises(ValueError):
        mw = ModelWrapper(model_to_fit=model_to_test_wrap,
                          fittable_params=1.0)
    with pytest.raises(ValueError):
        mw = ModelWrapper(model_to_fit=model_to_test_wrap,
                          fittable_params="a_string")
        
    # bad non_fit_kwargs
    with pytest.raises(ValueError):
        mw = ModelWrapper(model_to_fit=model_to_test_wrap,
                          non_fit_kwargs=[1.0])
    with pytest.raises(ValueError):
        mw = ModelWrapper(model_to_fit=model_to_test_wrap,
                          non_fit_kwargs="a_string")
        
    # bad default guess
    with pytest.raises(ValueError):
        mw = ModelWrapper(model_to_fit=model_to_test_wrap,
                          default_guess="not_a_number")

    # Test basic functionality. Makes sure model_to_fit being captured correctly
    mw = ModelWrapper(model_to_fit=model_to_test_wrap)
    assert mw._model_to_fit is model_to_test_wrap
    assert mw._default_guess == 0

    assert len(mw._param_df) == 3
    assert mw._param_df.loc["a","guess"] == 0
    assert mw._param_df.loc["b","guess"] == 2
    assert mw._param_df.loc["c","guess"] == 3
    
    assert len(mw._non_fit_kwargs) == 2
    assert mw._non_fit_kwargs["d"] == "test"
    assert mw._non_fit_kwargs["e"] == 3

    # make sure fittable_params are being passed 
    mw = ModelWrapper(model_to_test_wrap,
                      fittable_params=["a"])
    assert mw._model_to_fit is model_to_test_wrap
    assert mw._default_guess == 0

    assert len(mw._param_df) == 1
    assert mw._param_df.loc["a","guess"] == 0
    
    assert len(mw._non_fit_kwargs) == 4
    assert mw._non_fit_kwargs["b"] == 2
    assert mw._non_fit_kwargs["c"] == 3
    assert mw._non_fit_kwargs["d"] == "test"
    assert mw._non_fit_kwargs["e"] == 3

    # make sure non_fit_kwargs are being passed 
    mw = ModelWrapper(model_to_test_wrap,non_fit_kwargs={"a":20})
    assert mw._model_to_fit is model_to_test_wrap
    assert mw._default_guess == 0

    assert len(mw._param_df) == 2
    assert mw._param_df.loc["b","guess"] == 2
    assert mw._param_df.loc["c","guess"] == 3
    
    assert len(mw._non_fit_kwargs) == 3
    assert mw._non_fit_kwargs["a"] == 20
    assert mw._non_fit_kwargs["d"] == "test"
    assert mw._non_fit_kwargs["e"] == 3

    # make sure default_guess is being passed
    mw = ModelWrapper(model_to_test_wrap,default_guess=1.0)
    assert mw._model_to_fit is model_to_test_wrap
    assert mw._default_guess == 1

        

def test_ModelWrapper__load_model():

    # Create a ModelWrapper that has just been initialized but has not
    # run load_model
    class TestModelWrapper(ModelWrapper):
        def __init__(self):
            self._default_guess = 0.0
            self._param_df = pd.DataFrame({"name":[]})
            self._non_fit_kwargs = {}
            
    mw = TestModelWrapper()
    assert len(mw.__dict__) == 3

    def model_to_test_wrap(a,b=2,c=3,d="test",e=3): return a*b*c
    
    mw._load_model(model_to_test_wrap,
                   fittable_params=None,
                   non_fit_kwargs=None)
    assert mw._model_to_fit is model_to_test_wrap

    # analyze_fcn_sig, reconcile_fittable, param_sanity check are all tested in
    # test_function_processing. We can basically only test results here. The 
    # model above covers almost the whole decision tree. Tests of complete 
    # decision tree follow. 

    assert len(mw._param_df) == 3
    assert mw._param_df.loc["a","guess"] == 0
    assert mw._param_df.loc["b","guess"] == 2
    assert mw._param_df.loc["c","guess"] == 3
    
    assert len(mw._non_fit_kwargs) == 2
    assert mw._non_fit_kwargs["d"] == "test"
    assert mw._non_fit_kwargs["e"] == 3
    assert mw._non_fit_kwargs_keys == set(["d","e"])

    # This makes sure that the finalize_params() call is happening. only 
    # test here because the logic of that call is tested in its own method call. 
    assert np.array_equal(mw._fit_params_in_order,["a","b","c"])
    assert len(mw._mw_kwargs) == 5
    assert mw._mw_kwargs["a"] == 0.0
    assert mw._mw_kwargs["b"] == 2.0
    assert mw._mw_kwargs["c"] == 3.0
    assert mw._mw_kwargs["d"] == "test"
    assert mw._mw_kwargs["e"] == 3

    # Now validate interaction with input function and fittable_params. Only
    # grab one argument. 
    mw = TestModelWrapper()
    assert len(mw.__dict__) == 3
    mw._load_model(model_to_test_wrap,
                   fittable_params=["a"],
                   non_fit_kwargs=None)
    assert len(mw._param_df) == 1
    assert mw._param_df.loc["a","guess"] == 0
    
    assert len(mw._non_fit_kwargs) == 4
    assert mw._non_fit_kwargs["b"] == 2
    assert mw._non_fit_kwargs["c"] == 3
    assert mw._non_fit_kwargs["d"] == "test"
    assert mw._non_fit_kwargs["e"] == 3
    assert mw._non_fit_kwargs_keys == set(["b","c","d","e"])

    # Now validate interaction with input function and fittable_params. Add 
    # argument that would not normally be grabbed. 
    mw = TestModelWrapper()
    assert len(mw.__dict__) == 3
    mw._load_model(model_to_test_wrap,
                   fittable_params=["a","e"],
                   non_fit_kwargs=None)
    assert len(mw._param_df) == 2
    assert mw._param_df.loc["a","guess"] == 0
    assert mw._param_df.loc["e","guess"] == 3
    
    assert len(mw._non_fit_kwargs) == 3
    assert mw._non_fit_kwargs["b"] == 2
    assert mw._non_fit_kwargs["c"] == 3
    assert mw._non_fit_kwargs["d"] == "test"
    assert mw._non_fit_kwargs_keys == set(["b","c","d"])

    # Add argument not thought to be fittable by the parser.
    mw = TestModelWrapper()
    assert len(mw.__dict__) == 3
    with pytest.raises(ValueError):
        mw._load_model(model_to_test_wrap,
                       fittable_params=["a","d"],
                       non_fit_kwargs=None)

    # fittable param that is not in arguments
    mw = TestModelWrapper()
    assert len(mw.__dict__) == 3
    with pytest.raises(ValueError):
        mw._load_model(model_to_test_wrap,
                       fittable_params=["w"],
                       non_fit_kwargs=None)

    # not enough fittable params
    mw = TestModelWrapper()
    assert len(mw.__dict__) == 3
    with pytest.raises(ValueError):
        mw._load_model(model_to_test_wrap,
                       fittable_params=[],
                       non_fit_kwargs=None)
    
    # send in a model that is only kwargs and make sure it still gets a fittable
    # param.
    def model_to_test_wrap(**kwargs): return kwargs["a"]
    mw = TestModelWrapper()
    with pytest.raises(ValueError):
        mw._load_model(model_to_test_wrap,
                       fittable_params=None,
                       non_fit_kwargs=None)
        
    mw = TestModelWrapper()
    mw._load_model(model_to_test_wrap,
                   fittable_params=["a"],
                   non_fit_kwargs=None)
    assert len(mw._param_df) == 1
    assert mw._param_df.loc["a","guess"] == 0
    assert len(mw._non_fit_kwargs) == 0

    # Test non_fit_kwargs on non-fittable argument with no default
    def model_to_test_wrap(a,b=1,c="test"): return a*b
    mw = TestModelWrapper()
    mw._load_model(model_to_test_wrap,
                   fittable_params=None,
                   non_fit_kwargs={"a":42})
    assert len(mw._param_df) == 1
    assert mw._param_df.loc["b","guess"] == 1
    assert len(mw._non_fit_kwargs) == 2
    assert mw._non_fit_kwargs["a"] == 42
    assert mw._non_fit_kwargs["c"] == "test"
    assert mw._non_fit_kwargs_keys == set(["a","c"])
    
    # Test non_fit_kwargs on fittable argument with default
    mw = TestModelWrapper()
    mw._load_model(model_to_test_wrap,
                   fittable_params=None,
                   non_fit_kwargs={"b":17})
    assert len(mw._param_df) == 1
    assert mw._param_df.loc["a","guess"] == 0
    assert len(mw._non_fit_kwargs) == 2
    assert mw._non_fit_kwargs["b"] == 17
    assert mw._non_fit_kwargs["c"] == "test"
    assert mw._non_fit_kwargs_keys == set(["b","c"])

    # Test non_fit_kwargs on non-fittable argument with default
    mw = TestModelWrapper()
    mw._load_model(model_to_test_wrap,
                   fittable_params=None,
                   non_fit_kwargs={"c":(1,3)})
    assert len(mw._param_df) == 2
    assert mw._param_df.loc["a","guess"] == 0
    assert mw._param_df.loc["b","guess"] == 1
    assert len(mw._non_fit_kwargs) == 1
    assert mw._non_fit_kwargs["c"] == (1,3)
    assert mw._non_fit_kwargs_keys == set(["c"])

    # Test reasonable fittable_params and non_fit_kwargs
    mw = TestModelWrapper()
    mw._load_model(model_to_test_wrap,
                   fittable_params=["b"],
                   non_fit_kwargs={"a":10,
                                        "c":"rocket"})
    assert len(mw._param_df) == 1
    assert mw._param_df.loc["b","guess"] == 1
    assert len(mw._non_fit_kwargs) == 2
    assert mw._non_fit_kwargs["a"] == 10
    assert mw._non_fit_kwargs["c"] == "rocket"
    assert mw._non_fit_kwargs_keys == set(["a","c"])

    # Send in conflicting fittable and non_fit_kwargs
    mw = TestModelWrapper()
    with pytest.raises(ValueError):
        mw._load_model(model_to_test_wrap,
                       fittable_params=["a","b"],
                       non_fit_kwargs={"a":10})
    
    # Send in too many non_fit_kwargs
    mw = TestModelWrapper()
    with pytest.raises(ValueError):
        mw._load_model(model_to_test_wrap,
                       fittable_params=None,
                       non_fit_kwargs={"a":10,"b":20})
        
    # kwargs -- should take anything
    def model_to_test_wrap(**kwargs): pass
    mw = TestModelWrapper()
    mw._load_model(model_to_test_wrap,
                    fittable_params=["a","b"],
                    non_fit_kwargs={"c":5,"d":13})
    assert len(mw._param_df) == 2
    assert mw._param_df.loc["a","guess"] == 0
    assert mw._param_df.loc["b","guess"] == 0
    assert len(mw._non_fit_kwargs) == 2
    assert mw._non_fit_kwargs["c"] == 5
    assert mw._non_fit_kwargs["d"] == 13
    assert mw._non_fit_kwargs_keys == set(["c","d"])

    # kwargs -- should take anything, gracefully handle empty dict of non_fit_kwargs
    def model_to_test_wrap(**kwargs): pass
    mw = TestModelWrapper()
    mw._load_model(model_to_test_wrap,
                    fittable_params=["a","b"],
                    non_fit_kwargs={})
    assert len(mw._param_df) == 2
    assert mw._param_df.loc["a","guess"] == 0
    assert mw._param_df.loc["b","guess"] == 0
    assert len(mw._non_fit_kwargs) == 0
    assert mw._non_fit_kwargs_keys == set([])

    
def test_ModelWrapper__validate_non_fit_kwargs():

    def model_to_test_wrap(a=1,b=2,c=3,d="test",e=3): return a*b*c
    mw = ModelWrapper(model_to_test_wrap)
    assert mw._non_fit_kwargs_keys == set(["d","e"])

    # Set value, should work
    assert mw._non_fit_kwargs["e"] == 3
    mw.non_fit_kwargs["e"] = 5
    assert mw._non_fit_kwargs["e"] == 5
    mw._validate_non_fit_kwargs()

    # add extra value, should fail
    assert len(mw._non_fit_kwargs) == 2
    mw.non_fit_kwargs["f"] = 11
    with pytest.raises(ValueError):
        mw._validate_non_fit_kwargs()
    
    # remove needed value
    def model_to_test_wrap(a=1,b=2,c=3,d="test",e=3): return a*b*c
    mw = ModelWrapper(model_to_test_wrap)
    assert mw._non_fit_kwargs_keys == set(["d","e"])
    assert len(mw._non_fit_kwargs) == 2
    mw.non_fit_kwargs.pop("e")
    with pytest.raises(ValueError):
        mw._validate_non_fit_kwargs()


def test_ModelWrapper_finalize_params():

    def model_to_test_wrap(a=1,b=2,c=3,d="test",e=3): return a*b*c
    mw = ModelWrapper(model_to_test_wrap)
    
    # Check initial configuration after __init__
    assert np.array_equal(mw._fit_params_in_order,["a","b","c"])
    assert len(mw._non_fit_kwargs) == 2
    assert mw._non_fit_kwargs["d"] == "test"
    assert mw._non_fit_kwargs["e"] == 3
    assert len(mw._mw_kwargs) == 5
    assert mw._mw_kwargs["a"] == 1
    assert mw._mw_kwargs["b"] == 2
    assert mw._mw_kwargs["c"] == 3
    assert mw._mw_kwargs["d"] == "test"
    assert mw._mw_kwargs["e"] == 3

    # Edit parameters
    mw.param_df.loc["a","guess"] = 10
    mw.param_df.loc["a","fixed"] = True

    # Make sure no change
    assert np.array_equal(mw._fit_params_in_order,["a","b","c"])
    assert len(mw._non_fit_kwargs) == 2
    assert mw._non_fit_kwargs["d"] == "test"
    assert mw._non_fit_kwargs["e"] == 3
    assert len(mw._mw_kwargs) == 5
    assert mw._mw_kwargs["a"] == 1
    assert mw._mw_kwargs["b"] == 2
    assert mw._mw_kwargs["c"] == 3
    assert mw._mw_kwargs["d"] == "test"
    assert mw._mw_kwargs["e"] == 3
    assert np.array_equal(mw._unfixed_mask,[True,True,True])
    assert np.array_equal(mw._unfixed_param_names,["a","b","c"])

    # Run function
    mw.finalize_params()

    # Check for expected output
    assert np.array_equal(mw._fit_params_in_order,["a","b","c"])
    assert len(mw._non_fit_kwargs) == 2
    assert mw._non_fit_kwargs["d"] == "test"
    assert mw._non_fit_kwargs["e"] == 3
    assert len(mw._mw_kwargs) == 5
    assert mw._mw_kwargs["a"] == 10
    assert mw._mw_kwargs["b"] == 2
    assert mw._mw_kwargs["c"] == 3
    assert mw._mw_kwargs["d"] == "test"
    assert mw._mw_kwargs["e"] == 3
    assert np.array_equal(mw._unfixed_mask,[False,True,True])
    assert np.array_equal(mw._unfixed_param_names,["b","c"])
    
    # send in bad edit -- finalize should catch
    mw.param_df.loc["not_a_param","guess"] = 5
    assert np.array_equal(mw.param_df.index,["a","b","c","not_a_param"])
    with pytest.raises(ValueError):
        mw.finalize_params()

    # make sure it's running _validate_non_fit_kwargs
    def model_to_test_wrap(a=1,b=2,c=3,d="test",e=3): return a*b*c
    mw = ModelWrapper(model_to_test_wrap)
    
    # works fine
    mw.finalize_params()
    
    # add extra parameter
    mw.non_fit_kwargs["f"] = "extra_kwarg"
    with pytest.raises(ValueError):
        mw.finalize_params()

    # remove offending parameter
    mw.non_fit_kwargs.pop("f")
    mw.finalize_params()
    

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
    mw.param_df.loc["b","fixed"] = True
    mw.finalize_params()
    assert mw._mw_observable([3,4]) == 3*2*4 #(a*fixed(b)*c)

    # now fail because too many params
    with pytest.raises(ValueError):
        mw._mw_observable([3,4,5])

    def model_to_test_wrap(a=1,b=2,c=3,d="test",e=3): raise ValueError
    mw = ModelWrapper(model_to_test_wrap)
    with pytest.raises(RuntimeError):
        mw._mw_observable()

def test_ModelWrapper_update_params(spreadsheets):

    # method calls three other functions that are tested extensively elsewhere.
    # make sure they run, but do not test extensively. 

    # read from spreadsheet
    def model_to_test_wrap(K1=1,K2=2,K3=3,d="test",e=3): return K1*K2*K3
    mw = ModelWrapper(model_to_test_wrap)
    assert mw.param_df.loc["K1","guess"] == 1
    assert mw.param_df.loc["K2","guess"] == 2
    assert mw.param_df.loc["K3","guess"] == 3

    xlsx = spreadsheets["basic-spreadsheet.xlsx"]
    mw.update_params(xlsx)
    assert np.isclose(mw.param_df.loc["K1","guess"],1.00E+07)
    assert np.isclose(mw.param_df.loc["K2","guess"],1.00E-06)
    assert np.isclose(mw.param_df.loc["K3","guess"],1.00E+00)

    # read from df
    def model_to_test_wrap(K1=1,K2=2,K3=3,d="test",e=3): return K1*K2*K3
    mw = ModelWrapper(model_to_test_wrap)
    assert mw.param_df.loc["K1","guess"] == 1
    assert mw.param_df.loc["K2","guess"] == 2
    assert mw.param_df.loc["K3","guess"] == 3

    input_df = pd.read_excel(xlsx)
    mw.update_params(input_df)
    assert np.isclose(mw.param_df.loc["K1","guess"],1.00E+07)
    assert np.isclose(mw.param_df.loc["K2","guess"],1.00E-06)
    assert np.isclose(mw.param_df.loc["K3","guess"],1.00E+00)

    # read from dict
    def model_to_test_wrap(K1=1,K2=2,K3=3,d="test",e=3): return K1*K2*K3
    mw = ModelWrapper(model_to_test_wrap)
    assert mw.param_df.loc["K1","guess"] == 1
    assert mw.param_df.loc["K2","guess"] == 2
    assert mw.param_df.loc["K3","guess"] == 3

    input_dict = {"K1":{"guess":20}}
    mw.update_params(input_dict)
    assert mw.param_df.loc["K1","guess"] == 20
    assert mw.param_df.loc["K2","guess"] == 2
    assert mw.param_df.loc["K3","guess"] == 3

    # bad input
    def model_to_test_wrap(K1=1,K2=2,K3=3,d="test",e=3): return K1*K2*K3
    mw = ModelWrapper(model_to_test_wrap)
    assert mw.param_df.loc["K1","guess"] == 1
    assert mw.param_df.loc["K2","guess"] == 2
    assert mw.param_df.loc["K3","guess"] == 3

    input_dict = {"K1":20}
    with pytest.raises(ValueError):
        mw.update_params(input_dict)

    # bad input
    def model_to_test_wrap(K1=1,K2=2,K3=3,d="test",e=3): return K1*K2*K3
    mw = ModelWrapper(model_to_test_wrap)
    input_dict = {"K4":20}
    with pytest.raises(ValueError):
        mw.update_params(input_dict)

    # inconsistent input
    def model_to_test_wrap(K1=1,K2=2,K3=3,d="test",e=3): return K1*K2*K3
    mw = ModelWrapper(model_to_test_wrap)
    input_dict = {"K1":{"guess":5,
                        "lower_bound":20}}
    with pytest.raises(ValueError):
        mw.update_params(input_dict)
    
def test_ModelWrapper_model():

    # light wrapper for _mw_observable (tested elsewhere). Make sure finalize
    # runs and that it works as advertised but do not test deeply

    def model_to_test_wrap(a=1,b=2,c=3,d="test",e=3): return a*b*c
    mw = ModelWrapper(model_to_test_wrap)
    m = mw.model
    
    # Make sure it is callable and takes arguments
    assert hasattr(m,"__call__")
    assert m() == 1*2*3
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

    # but model is because it calls finalize
    with pytest.raises(ValueError):
        assert mw.model([1,2,3])
    assert mw.model([2,3]) == 10*2*3

    # and now mw_observable should be too
    assert mw._mw_observable([2,3]) == 10*2*3
    
def test_ModelWrapper_fast_model():

    # light wrapper for _mw_observable (tested elsewhere). Make sure finalize
    # runs and that it works as advertised but do not test deeply

    def model_to_test_wrap(a=1,b=2,c=3,d="test",e=3): return a*b*c
    mw = ModelWrapper(model_to_test_wrap)
    
    # Make sure it is callable and takes arguments
    assert mw.fast_model([10,20,30]) == 10*20*30

    # Make sure it calls finalize. 
    # Run model and fast_model
    assert mw.fast_model([1,2,3]) == 1*2*3
    assert mw._mw_observable([1,2,3]) == 1*2*3

    # fix and change "a"
    mw.param_df.loc["a","fixed"] = True
    mw.param_df.loc["a","guess"] = 10

    # fast_model is not aware of this 
    assert mw.fast_model([1,2,3]) == 1*2*3

    mw.finalize_params()

    # but now fast_model is because we finalized parameters
    assert mw.fast_model([2,3]) == 10*2*3


def test_ModelWrapper_param_df():

    # test setter/getter
    
    def model_to_test_wrap(a=1,b=2,c=3,d="test",e=3): return a*b*c
    mw = ModelWrapper(model_to_test_wrap)
    
    assert mw._param_df is mw.param_df
    assert np.array_equal(mw.param_df["name"],["a","b","c"])
    
    to_change = mw.param_df.copy()
    to_change["guess"] = [10,20,30]
    mw.param_df = to_change
    assert np.array_equal(mw.param_df["guess"],[10,20,30])
    assert mw.param_df is not to_change

    # should fail vaildation because parameter name changed
    to_change["name"] = ["x","b","c"]
    with pytest.raises(ValueError):
        mw.param_df = to_change

    # Make sure it didn't change anything on failure
    assert np.array_equal(mw.param_df["name"],["a","b","c"])    
    assert np.array_equal(mw.param_df["guess"],[10,20,30])

def test_ModelWrapper_non_fit_kwargs():

    def model_to_test_wrap(a=1,b=2,c=3,d="test",e=3): return a*b*c
    mw = ModelWrapper(model_to_test_wrap)
    assert mw._non_fit_kwargs is mw.non_fit_kwargs
    assert mw.non_fit_kwargs["d"] == "test"
    assert mw.non_fit_kwargs["e"] == 3

def test_ModelWrapper_unfixed_mask():
    
    def model_to_test_wrap(a=1,b=2,c=3,d="test",e=3): return a*b*c
    mw = ModelWrapper(model_to_test_wrap)
    assert np.array_equal(mw.unfixed_mask,[True,True,True])
    
    # set to fixed -- should not update until finalized
    mw.param_df.loc["a","fixed"] = True
    assert np.array_equal(mw.unfixed_mask,[True,True,True])
    mw.finalize_params()
    assert np.array_equal(mw.unfixed_mask,[False,True,True])


def test_ModelWrapper___repr__():

    def model_to_test_wrap(a=1,b=1,c="test",d=3): return a*b

    mw = ModelWrapper(model_to_fit=model_to_test_wrap)

    out = mw.__repr__().split("\n")
    assert len(out) == 21

    assert out[0] == "ModelWrapper"
    assert out[1] == "------------"

    # This will force truncated variable lines to run by making super huge
    # c fixed argument
    mw.non_fit_kwargs["c"] = pd.DataFrame({"out":np.arange(1000)})
    out = mw.__repr__().split("\n")
    assert len(out) == 27

    
    
