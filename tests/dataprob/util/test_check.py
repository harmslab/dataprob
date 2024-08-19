
from dataprob.util.check import check_bool
from dataprob.util.check import check_float
from dataprob.util.check import check_int
from dataprob.util.check import check_array

import pytest
import numpy as np
import pandas as pd

def test_check_bool():

    true_values = [True,1.0,1,np.ones(1,dtype=np.bool_)[0],1.000001]
    for t in true_values:
        assert check_bool(t)

    false_values = [False,0.0,0,np.zeros(1,dtype=np.bool_)[0]]
    for f in false_values:
        assert not check_bool(f)

    bad_value = [None,123,-1,bool,"stupid",[1.0,1.0],np.array([1.0,1.0]),{},float,1.1,0.1]
    for b in bad_value:
        with pytest.raises(ValueError):
            value = check_bool(b)


def test_check_float():

    value = check_float(1.0)
    assert value == 1.0

    bad_value = [None,"stupid",[1.0,1.0],np.array([1.0,1.0]),{},float,np.nan]
    for b in bad_value:
        print("bad value", b)
        with pytest.raises(ValueError):
            value = check_float(b)

    good_value = [-np.inf,np.inf,0,1,"1.0"]
    for g in good_value:
        value = check_float(g)

    with pytest.raises(ValueError):
        check_float(1.0,minimum_allowed=2.0)

    with pytest.raises(ValueError):
        check_float(1.0,minimum_allowed=1.0,minimum_inclusive=False)

    value = check_float(1.0,minimum_allowed=1.0,minimum_inclusive=True)
    assert value == 1

    with pytest.raises(ValueError):
        check_float(1.0,maximum_allowed=0.5)

    with pytest.raises(ValueError):
        check_float(1.0,maximum_allowed=1.0,maximum_inclusive=False)

    value = check_float(1.0,minimum_allowed=1.0,maximum_inclusive=True)
    assert value == 1

    with pytest.raises(ValueError):
        check_float(np.nan)
    assert np.isnan(check_float(np.nan,allow_nan=True))
    
    with pytest.raises(ValueError):
        check_float(pd.NA)
    assert np.isnan(check_float(pd.NA,allow_nan=True))

    with pytest.raises(ValueError):
        check_float(None)
    assert np.isnan(check_float(None,allow_nan=True))

    # check bounds == None case for error message
    with pytest.raises(ValueError):
        check_float(None,minimum_allowed=None,maximum_allowed=None)



def test_check_int():

    value = check_int(1)
    assert value == 1

    bad_value = [None,"stupid",[1.0,1.0],np.array([1.0,1.0]),{},float,int,np.inf,np.nan,1.3]
    for b in bad_value:
        print(b)
        with pytest.raises(ValueError):
            value = check_int(b)

    good_value = [-10,0,10,"10",10.0]
    for g in good_value:
        value = check_int(g)

    with pytest.raises(ValueError):
        check_int(1,minimum_allowed=2.0)

    with pytest.raises(ValueError):
        check_int(1,minimum_allowed=1,minimum_inclusive=False)
    
    value = check_int(1,minimum_allowed=1,minimum_inclusive=True)
    assert value == 1
    value = check_int(2,minimum_allowed=1,minimum_inclusive=True)
    assert value == 2
    value = check_int(2,minimum_allowed=1,minimum_inclusive=False)
    assert value == 2

    with pytest.raises(ValueError):
        check_int(1,maximum_allowed=0)

    with pytest.raises(ValueError):
        check_int(1,maximum_allowed=1,maximum_inclusive=False)

    value = check_int(1,maximum_allowed=1,maximum_inclusive=True)
    assert value == 1
    value = check_int(0,maximum_allowed=1,maximum_inclusive=True)
    assert value == 0
    value = check_int(0,maximum_allowed=1,maximum_inclusive=False)
    assert value == 0

def test_check_array():
    
    for variable_name in [None,"variable_name"]:

        print("variable_name",variable_name)
    
        # Stuff without iter that could go in 
        bad_value = [None,1.0,float,np.nan]
        for b in bad_value:
            print("bad value",b)
            with pytest.raises(ValueError):
                check_array(b,variable_name=variable_name)

        # stuff with iter that can't be coerced to float
        bad_value = ["test",{"x":1},{"x":[1,2]},["a",1.0]]
        for b in bad_value:
            print("bad value",b)
            with pytest.raises(ValueError):
                check_array(b,variable_name=variable_name)

        # stuff that should be fine 
        good_value = [[1.0],[1],[np.ones(1)[0]],np.ones(1)]
        for g in good_value:
            print("good value",g)
            value = check_array(g,variable_name=variable_name)
            assert np.array_equal;(value,g)

        for expected_shape_name in [None,"expected_shape_name"]:

            print("expected_shape_name",expected_shape_name)

            # Should be fine with dimensions specified
            good_value = [[1.0],[1],[np.ones(1)[0]],np.ones(1)]
            for g in good_value:
                print('good value',g)
                value = check_array(g,
                                    variable_name=g,
                                    expected_shape=(1,),
                                    expected_shape_names=expected_shape_name)
                assert np.array_equal(value,g)

            # Should fail now with dimensions specified
            good_value = [[1.0],[1],[np.ones(1)[0]],np.ones(1)]
            for g in good_value:
                print('good value, now bad',g)
                with pytest.raises(ValueError):
                    check_array(g,
                                variable_name=g,
                                expected_shape=(1,2),
                                expected_shape_names=expected_shape_name)
                          
            # right number of dimensions, wrong length
            good_value = [[1.0],[1],[np.ones(1)[0]],np.ones(1)]
            for g in good_value:
                print('good value, now bad',g)
                with pytest.raises(ValueError):
                    check_array(g,
                                variable_name=g,
                                expected_shape=(10,),
                                expected_shape_names=expected_shape_name)
            
            # good, 2D but no specific lengths
            good_value = [[[1.0,1.0,1.0],
                           [1.0,1.0,1.0]],
                           np.ones((2,3),dtype=float)]
            for g in good_value:
                print('good value',g)
                value = check_array(g,
                                    variable_name=g,
                                    expected_shape=(None,None),
                                    expected_shape_names=expected_shape_name)
                assert np.array_equal(value,g)

            # good, 2D but specific first length
            good_value = [[[1.0,1.0,1.0],
                           [1.0,1.0,1.0]],
                           np.ones((2,3),dtype=float)]
            for g in good_value:
                print('good value',g)
                value = check_array(g,
                                    variable_name=g,
                                    expected_shape=(2,None),
                                    expected_shape_names=expected_shape_name)
                assert np.array_equal(value,g)

            # good, 2D but specific second length
            good_value = [[[1.0,1.0,1.0],
                           [1.0,1.0,1.0]],
                           np.ones((2,3),dtype=float)]
            for g in good_value:
                print('good value',g)
                value = check_array(g,
                                    variable_name=g,
                                    expected_shape=(None,3),
                                    expected_shape_names=expected_shape_name)
                assert np.array_equal(value,g)


            # good, specific both lengths
            good_value = [[[1.0,1.0,1.0],
                           [1.0,1.0,1.0]],
                           np.ones((2,3),dtype=float)]
            for g in good_value:
                print('good value',g)
                value = check_array(g,
                                    variable_name=g,
                                    expected_shape=(2,3),
                                    expected_shape_names=expected_shape_name)
                assert np.array_equal(value,g)


            # bad lengths
            shapes = [(3,3),(3,None),(None,2)]
            for s in shapes:
                print("shape",s)
                bad_value = [[[1.0,1.0,1.0],
                            [1.0,1.0,1.0]],
                            np.ones((2,3),dtype=float)]
                for b in bad_value:
                    print('bad_value',g)
                    with pytest.raises(ValueError):
                        check_array(g,
                                    variable_name=g,
                                    expected_shape=s,
                                    expected_shape_names=expected_shape_name)

            # nans
            has_nan = np.array([1,2,np.nan])
            v = check_array(has_nan,nan_allowed=True)
            with pytest.raises(ValueError):
                check_array(has_nan,nan_allowed=False)

            # check a couple of other possible nan inputs...
            v = check_array([None,1],nan_allowed=True)
            with pytest.raises(ValueError):
                check_array([None,1],nan_allowed=False)
            
            v = check_array([pd.NA,1],nan_allowed=True)
            with pytest.raises(ValueError):
                check_array([pd.NA,1],nan_allowed=False)


