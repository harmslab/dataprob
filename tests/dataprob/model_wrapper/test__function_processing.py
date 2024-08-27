
import pytest

from dataprob.model_wrapper._function_processing import analyze_fcn_sig
from dataprob.model_wrapper._function_processing import reconcile_fittable
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

    # only kwargs
    def test_fcn(**kwargs): pass
    all_args, can_be_fit, cannot_be_fit, has_kwargs = analyze_fcn_sig(test_fcn)
    assert len(all_args) == 0
    assert len(can_be_fit) == 0
    assert len(cannot_be_fit) == 0
    assert has_kwargs is True

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

    # Send in a list
    def test_fcn(a=[1,2,3]): pass
    
    all_args, can_be_fit, cannot_be_fit, has_kwargs = analyze_fcn_sig(test_fcn)
    assert np.array_equal(all_args,["a"])
    assert len(can_be_fit) == 0
    assert len(cannot_be_fit) == 1
    assert np.array_equal(cannot_be_fit["a"],[1,2,3])
    assert has_kwargs is False

    # Send in a float numpy array
    def test_fcn(a=np.array([1,2,3])): pass

    all_args, can_be_fit, cannot_be_fit, has_kwargs = analyze_fcn_sig(test_fcn)
    assert np.array_equal(all_args,["a"])
    assert len(can_be_fit) == 0
    assert len(cannot_be_fit) == 1
    assert np.array_equal(cannot_be_fit["a"],[1,2,3])
    assert has_kwargs is False

    # test for bool default not being fittable
    def test_fcn(a=1,b=False): pass
    all_args, can_be_fit, cannot_be_fit, has_kwargs = analyze_fcn_sig(test_fcn)
    assert len(all_args) == 2
    assert len(can_be_fit) == 1
    assert len(cannot_be_fit) == 1
    assert has_kwargs is False
    assert can_be_fit["a"] == 1.0
    assert cannot_be_fit["b"] is False

def test_reconcile_fittable():

    base_kwargs = {"fit_parameters":None,
                   "non_fit_kwargs":None,
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
    kwargs["fit_parameters"] = ["a"]
    kwargs["all_args"] = ["a"]
    kwargs["cannot_be_fit"] = {"a":"test"}
    with pytest.raises(ValueError):
        fittable, not_fittable = reconcile_fittable(**kwargs)
    
    # No fittable parameter. tell it to use "b" (not in function, no kwarg)
    kwargs = copy.deepcopy(base_kwargs)
    kwargs["fit_parameters"] = ["b"]
    kwargs["all_args"] = ["a"]
    kwargs["cannot_be_fit"] = {"a":"test"}
    with pytest.raises(ValueError):
        fittable, not_fittable = reconcile_fittable(**kwargs)

    # No fittable parameter, but kwarg. tell it to use "b"
    kwargs = copy.deepcopy(base_kwargs)
    kwargs["fit_parameters"] = ["b"]
    kwargs["all_args"] = ["a"]
    kwargs["cannot_be_fit"] = {"a":"test"}
    kwargs["has_kwargs"] = True
    fittable, not_fittable = reconcile_fittable(**kwargs)
    assert np.array_equal(fittable,["b"])
    assert np.array_equal(not_fittable,["a"])

    # Make sure it trims off after first fittable arg
    kwargs = copy.deepcopy(base_kwargs)
    kwargs["fit_parameters"] = None
    kwargs["all_args"] = ["a","b","c"]
    kwargs["can_be_fit"] = {"a":1,"c":2}
    kwargs["cannot_be_fit"] = {"b":"test"}
    kwargs["has_kwargs"] = False
    fittable, not_fittable = reconcile_fittable(**kwargs)
    assert np.array_equal(fittable,["a"])
    assert np.array_equal(not_fittable,["b","c"])

    # Make sure we can force it to take after non-fittable breaks arg list
    kwargs = copy.deepcopy(base_kwargs)
    kwargs["fit_parameters"] = ["a","c"]
    kwargs["all_args"] = ["a","b","c"]
    kwargs["can_be_fit"] = {"a":1,"c":2}
    kwargs["cannot_be_fit"] = {"b":"test"}
    kwargs["has_kwargs"] = False
    fittable, not_fittable = reconcile_fittable(**kwargs)
    assert np.array_equal(fittable,["a","c"])
    assert np.array_equal(not_fittable,["b"])

    # Fail. non-fittable in the mix
    kwargs = copy.deepcopy(base_kwargs)
    kwargs["fit_parameters"] = ["a","b"]
    kwargs["all_args"] = ["a","b","c"]
    kwargs["can_be_fit"] = {"a":1,"c":2}
    kwargs["cannot_be_fit"] = {"b":"test"}
    kwargs["has_kwargs"] = False
    with pytest.raises(ValueError):
        fittable, not_fittable = reconcile_fittable(**kwargs)

    # Fail. non-fittable in the mix
    kwargs = copy.deepcopy(base_kwargs)
    kwargs["fit_parameters"] = ["a","d"]
    kwargs["all_args"] = ["a","b","c","d"]
    kwargs["can_be_fit"] = {"a":1,"c":2}
    kwargs["cannot_be_fit"] = {"b":"test"}
    kwargs["has_kwargs"] = True
    
    fittable, not_fittable = reconcile_fittable(**kwargs)
    assert np.array_equal(fittable,["a","d"])
    assert np.array_equal(not_fittable,["b","c"])

    # Send in non_fit_kwargs where this is in "cannot be fit"
    kwargs = copy.deepcopy(base_kwargs)
    kwargs["fit_parameters"] = ["a","b"]
    kwargs["non_fit_kwargs"] = {"c":33}
    kwargs["all_args"] = ["a","b","c"]
    kwargs["can_be_fit"] = {"a":1,"b":2}
    kwargs["cannot_be_fit"] = {"c":"test"}
    kwargs["has_kwargs"] = True

    fittable, not_fittable = reconcile_fittable(**kwargs)
    assert np.array_equal(fittable,["a","b"])
    assert np.array_equal(not_fittable,["c"])
    
    # Send in non_fit_kwargs where this is in "can be fit"
    kwargs = copy.deepcopy(base_kwargs)
    kwargs["fit_parameters"] = ["a","b"]
    kwargs["non_fit_kwargs"] = {"c":33}
    kwargs["all_args"] = ["a","b","c"]
    kwargs["can_be_fit"] = {"a":1,"b":2,"c":3}
    kwargs["cannot_be_fit"] = {}
    kwargs["has_kwargs"] = True
    fittable, not_fittable = reconcile_fittable(**kwargs)
    assert np.array_equal(fittable,["a","b"])
    assert np.array_equal(not_fittable,["c"])

    # Send in non_fit_kwargs but not fit_parameters
    kwargs = copy.deepcopy(base_kwargs)
    kwargs["fit_parameters"] = None
    kwargs["non_fit_kwargs"] = {"c":33}
    kwargs["all_args"] = ["a","b","c"]
    kwargs["can_be_fit"] = {"a":1,"b":2,"c":3}
    kwargs["cannot_be_fit"] = {}
    kwargs["has_kwargs"] = True
    fittable, not_fittable = reconcile_fittable(**kwargs)
    assert np.array_equal(fittable,["a","b"])
    assert np.array_equal(not_fittable,["c"])

    # Send in same value in non_fit_kwargs and fit_parameters
    kwargs = copy.deepcopy(base_kwargs)
    kwargs["fit_parameters"] = ["a","b"]
    kwargs["non_fit_kwargs"] = {"b":None,"c":None}
    kwargs["all_args"] = ["a","b","c"]
    kwargs["can_be_fit"] = {"a":1,"b":2,"c":3}
    kwargs["cannot_be_fit"] = {}
    kwargs["has_kwargs"] = True
    with pytest.raises(ValueError):
        fittable, not_fittable = reconcile_fittable(**kwargs)

    # Send in all args as non_fit_kwargs, not fittable!
    kwargs = copy.deepcopy(base_kwargs)
    kwargs["fit_parameters"] = None
    kwargs["non_fit_kwargs"] = {"a":1,"b":3,"c":3}
    kwargs["all_args"] = ["a","b","c"]
    kwargs["can_be_fit"] = {"a":1,"b":2,"c":3}
    kwargs["cannot_be_fit"] = {}
    kwargs["has_kwargs"] = True
    with pytest.raises(ValueError):
        fittable, not_fittable = reconcile_fittable(**kwargs)
    
    # Send in non_fit_kwargs  that are not in the function signature
    kwargs = copy.deepcopy(base_kwargs)
    kwargs["fit_parameters"] = None
    kwargs["non_fit_kwargs"] = {"d":None}
    kwargs["all_args"] = ["a","b","c"]
    kwargs["can_be_fit"] = {"a":1,"b":2,"c":3}
    kwargs["cannot_be_fit"] = {}
    kwargs["has_kwargs"] = False
    with pytest.raises(ValueError):
        fittable, not_fittable = reconcile_fittable(**kwargs)

    # Send in non_fit_kwargs that are not in the function signature, but
    # has_kwargs, so okay
    kwargs = copy.deepcopy(base_kwargs)
    kwargs["fit_parameters"] = None
    kwargs["non_fit_kwargs"] = {"d":None}
    kwargs["all_args"] = ["a","b","c"]
    kwargs["can_be_fit"] = {"a":1,"b":2,"c":3}
    kwargs["cannot_be_fit"] = {}
    kwargs["has_kwargs"] = True
    
    fittable, not_fittable = reconcile_fittable(**kwargs)
    assert np.array_equal(fittable,["a","b","c"])
    assert np.array_equal(not_fittable,["d"])


    # Send in fit_parameters and non_fit_kwargs with only **kwargs
    kwargs = copy.deepcopy(base_kwargs)
    kwargs["fit_parameters"] = ["a","b"]
    kwargs["non_fit_kwargs"] = {"c":None,"d":None}
    kwargs["all_args"] = []
    kwargs["can_be_fit"] = {}
    kwargs["cannot_be_fit"] = {}
    kwargs["has_kwargs"] = True
    
    fittable, not_fittable = reconcile_fittable(**kwargs)
    assert np.array_equal(fittable,["a","b"])
    assert np.array_equal(not_fittable,["c","d"])


def test_analyze_vector_input_fcn():

    def test_fcn(a,b=1,*args,**kwargs): pass
    
    first_args, other_kwargs, has_kwargs = analyze_vector_input_fcn(test_fcn)
    assert first_args == "a"
    assert len(other_kwargs) == 1
    assert other_kwargs["b"] == 1
    assert has_kwargs is True

    def test_fcn(*args,**kwargs): pass
    
    first_args, other_kwargs, has_kwargs = analyze_vector_input_fcn(test_fcn)
    assert first_args == None
    assert len(other_kwargs) == 0
    assert has_kwargs is True

    def test_fcn(a=20,*args,**kwargs): pass
    
    first_args, other_kwargs, has_kwargs = analyze_vector_input_fcn(test_fcn)
    assert first_args == "a"
    assert len(other_kwargs) == 0
    assert has_kwargs is True

    def test_fcn(a,b,c,d=5,*args,**kwargs): pass
    
    first_args, other_kwargs, has_kwargs = analyze_vector_input_fcn(test_fcn)
    assert first_args == "a"
    assert len(other_kwargs) == 3
    assert other_kwargs["b"] is None
    assert other_kwargs["c"] is None
    assert other_kwargs["d"] == 5
    assert has_kwargs is True

    def test_fcn(a,b,c,d=5): pass
    
    first_args, other_kwargs, has_kwargs = analyze_vector_input_fcn(test_fcn)
    assert first_args == "a"
    assert len(other_kwargs) == 3
    assert other_kwargs["b"] is None
    assert other_kwargs["c"] is None
    assert other_kwargs["d"] == 5
    assert has_kwargs is False

    def test_fcn(a,b,c,d,*args): pass
    
    first_args, other_kwargs, has_kwargs= analyze_vector_input_fcn(test_fcn)
    assert first_args == "a"
    assert len(other_kwargs) == 3
    assert other_kwargs["b"] is None
    assert other_kwargs["c"] is None
    assert other_kwargs["d"] is None
    assert has_kwargs is False
    