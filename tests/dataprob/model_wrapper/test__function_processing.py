
import pytest

from dataprob.model_wrapper._function_processing import analyze_fcn_sig
from dataprob.model_wrapper._function_processing import reconcile_fittable
from dataprob.model_wrapper._function_processing import param_sanity_check
from dataprob.model_wrapper._function_processing import analyze_vector_input_fcn

import numpy as np

import copy

def test_analyze_fcn_sig():

    def test_fcn(a,b=2,c="test",d=3,*args,**kwargs): pass
    
    all_args, can_be_fit, cannot_be_fit, has_kwargs = analyze_fcn_sig(test_fcn)
    assert np.array_equal(all_args,["a","b","c","d"])
    assert len(can_be_fit) == 3
    assert can_be_fit["a"] is None
    assert can_be_fit["b"] == 2
    assert can_be_fit["d"] == 3
    assert len(cannot_be_fit) == 1
    assert cannot_be_fit["c"] == "test"
    assert has_kwargs is True

    # Drop kwargs
    def test_fcn(a,b=2,c="test",d=3,*args): pass
    
    all_args, can_be_fit, cannot_be_fit, has_kwargs = analyze_fcn_sig(test_fcn)
    assert np.array_equal(all_args,["a","b","c","d"])
    assert len(can_be_fit) == 3
    assert can_be_fit["a"] is None
    assert can_be_fit["b"] == 2
    assert can_be_fit["d"] == 3
    assert len(cannot_be_fit) == 1
    assert cannot_be_fit["c"] == "test"
    assert has_kwargs is False

    # Drop args, but keep kwargs
    def test_fcn(a,b=2,c="test",d=3,**kwargs): pass
    
    all_args, can_be_fit, cannot_be_fit, has_kwargs = analyze_fcn_sig(test_fcn)
    assert np.array_equal(all_args,["a","b","c","d"])
    assert len(can_be_fit) == 3
    assert can_be_fit["a"] is None
    assert can_be_fit["b"] == 2
    assert can_be_fit["d"] == 3
    assert len(cannot_be_fit) == 1
    assert cannot_be_fit["c"] == "test"
    assert has_kwargs is True

    # Nothing
    def test_fcn(): pass
    
    all_args, can_be_fit, cannot_be_fit, has_kwargs = analyze_fcn_sig(test_fcn)
    assert len(all_args) == 0
    assert len(can_be_fit) == 0
    assert len(cannot_be_fit) == 0
    assert has_kwargs is False

    # one fittable arg, no default
    def test_fcn(a): pass
    
    all_args, can_be_fit, cannot_be_fit, has_kwargs = analyze_fcn_sig(test_fcn)
    assert np.array_equal(all_args,["a"])
    assert len(can_be_fit) == 1
    assert can_be_fit["a"] is None
    assert len(cannot_be_fit) == 0
    assert has_kwargs is False

    # one fittable arg, no default
    def test_fcn(a): pass
    
    all_args, can_be_fit, cannot_be_fit, has_kwargs = analyze_fcn_sig(test_fcn)
    assert np.array_equal(all_args,["a"])
    assert len(can_be_fit) == 1
    assert can_be_fit["a"] is None
    assert len(cannot_be_fit) == 0
    assert has_kwargs is False

    # one non-fittable arg
    def test_fcn(a="test"): pass
    
    all_args, can_be_fit, cannot_be_fit, has_kwargs = analyze_fcn_sig(test_fcn)
    assert np.array_equal(all_args,["a"])
    assert len(can_be_fit) == 0
    assert len(cannot_be_fit) == 1
    assert cannot_be_fit["a"] == "test"
    assert has_kwargs is False

def test_reconcile_fittable():

    base_kwargs = {"fittable_params":None,
                   "all_args":[],
                   "can_be_fit":{},
                   "cannot_be_fit":{},
                   "has_kwargs":False}


    # no parameters at all
    kwargs = copy.deepcopy(base_kwargs)
    with pytest.raises(ValueError):
        fittable, not_fittable = reconcile_fittable(**kwargs)

    # No fittable parameters
    kwargs = copy.deepcopy(base_kwargs)
    kwargs["all_args"] = ["a"]
    kwargs["cannot_be_fit"] = {"a":"test"}
    with pytest.raises(ValueError):
        fittable, not_fittable = reconcile_fittable(**kwargs)

    # No fittable parameter, but we tell function to use "a" anyway
    kwargs = copy.deepcopy(base_kwargs)
    kwargs["fittable_params"] = ["a"]
    kwargs["all_args"] = ["a"]
    kwargs["cannot_be_fit"] = {"a":"test"}
    with pytest.raises(ValueError):
        fittable, not_fittable = reconcile_fittable(**kwargs)
    
    # No fittable parameter. tell it to use "b" (not in function, no kwarg)
    kwargs = copy.deepcopy(base_kwargs)
    kwargs["fittable_params"] = ["b"]
    kwargs["all_args"] = ["a"]
    kwargs["cannot_be_fit"] = {"a":"test"}
    with pytest.raises(ValueError):
        fittable, not_fittable = reconcile_fittable(**kwargs)

    # No fittable parameter, but kwarg. tell it to use "b"
    kwargs = copy.deepcopy(base_kwargs)
    kwargs["fittable_params"] = ["b"]
    kwargs["all_args"] = ["a"]
    kwargs["cannot_be_fit"] = {"a":"test"}
    kwargs["has_kwargs"] = True
    fittable, not_fittable = reconcile_fittable(**kwargs)
    assert np.array_equal(fittable,["b"])
    assert np.array_equal(not_fittable,["a"])

    # Make sure it trims off after first fittable arg
    kwargs = copy.deepcopy(base_kwargs)
    kwargs["fittable_params"] = None
    kwargs["all_args"] = ["a","b","c"]
    kwargs["can_be_fit"] = {"a":1,"c":2}
    kwargs["cannot_be_fit"] = {"b":"test"}
    kwargs["has_kwargs"] = False
    fittable, not_fittable = reconcile_fittable(**kwargs)
    assert np.array_equal(fittable,["a"])
    assert np.array_equal(not_fittable,["b","c"])

    # Make sure we can force it to take after non-fittable breaks arg list
    kwargs = copy.deepcopy(base_kwargs)
    kwargs["fittable_params"] = ["a","c"]
    kwargs["all_args"] = ["a","b","c"]
    kwargs["can_be_fit"] = {"a":1,"c":2}
    kwargs["cannot_be_fit"] = {"b":"test"}
    kwargs["has_kwargs"] = False
    fittable, not_fittable = reconcile_fittable(**kwargs)
    assert np.array_equal(fittable,["a","c"])
    assert np.array_equal(not_fittable,["b"])

    # Fail. non-fittable in the mix
    kwargs = copy.deepcopy(base_kwargs)
    kwargs["fittable_params"] = ["a","b"]
    kwargs["all_args"] = ["a","b","c"]
    kwargs["can_be_fit"] = {"a":1,"c":2}
    kwargs["cannot_be_fit"] = {"b":"test"}
    kwargs["has_kwargs"] = False
    with pytest.raises(ValueError):
        fittable, not_fittable = reconcile_fittable(**kwargs)

    # Fail. non-fittable in the mix
    kwargs = copy.deepcopy(base_kwargs)
    kwargs["fittable_params"] = ["a","d"]
    kwargs["all_args"] = ["a","b","c"]
    kwargs["can_be_fit"] = {"a":1,"c":2}
    kwargs["cannot_be_fit"] = {"b":"test"}
    kwargs["has_kwargs"] = True
    
    fittable, not_fittable = reconcile_fittable(**kwargs)
    assert np.array_equal(fittable,["a","d"])
    assert np.array_equal(not_fittable,["b","c"])
    
def test_param_sanity_check():

    result = param_sanity_check(fittable_params=["a","b"],
                                 reserved_params=None)
    assert np.array_equal(result,["a","b"])

    with pytest.raises(ValueError):
        result = param_sanity_check(fittable_params=["a","b"],
                                     reserved_params=["a"])
    
    result = param_sanity_check(fittable_params=[],
                                 reserved_params=None)
    assert np.array_equal(result,[])

    result = param_sanity_check(fittable_params=[],
                                 reserved_params=["a","b"])
    assert np.array_equal(result,[])

def test_analyze_vector_input_fcn():

    def test_fcn(a,b=1,*args,**kwargs): pass
    
    first_args, other_kwargs = analyze_vector_input_fcn(test_fcn)
    assert first_args == "a"
    assert len(other_kwargs) == 1
    assert other_kwargs["b"] == 1

    def test_fcn(*args,**kwargs): pass
    
    first_args, other_kwargs = analyze_vector_input_fcn(test_fcn)
    assert first_args == None
    assert len(other_kwargs) == 0

    def test_fcn(a=20,*args,**kwargs): pass
    
    first_args, other_kwargs = analyze_vector_input_fcn(test_fcn)
    assert first_args == "a"
    assert len(other_kwargs) == 0

    def test_fcn(a,b,c,d=5,*args,**kwargs): pass
    
    first_args, other_kwargs = analyze_vector_input_fcn(test_fcn)
    assert first_args == "a"
    assert len(other_kwargs) == 3
    assert other_kwargs["b"] is None
    assert other_kwargs["c"] is None
    assert other_kwargs["d"] == 5

    