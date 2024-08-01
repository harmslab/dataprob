
from dataprob.check import check_bool
from dataprob.check import check_float
from dataprob.check import check_int
from dataprob.check import column_to_bool

import pytest
import numpy as np
import pandas as pd

def test_check_bool():

    true_values = [True,1.0,1,np.ones(1,dtype=np.bool_)[0]]
    for t in true_values:
        assert check_bool(t)

    false_values = [False,0.0,0,np.zeros(1,dtype=np.bool_)[0]]
    for f in false_values:
        assert not check_bool(f)

    bad_value = [None,123,-1,bool,"stupid",[1.0,1.0],np.array([1.0,1.0]),{},float]
    for b in bad_value:
        with pytest.raises(ValueError):
            value = check_bool(b)


def test_check_float():

    value = check_float(1.0)
    assert value == 1.0

    bad_value = [None,"stupid",[1.0,1.0],np.array([1.0,1.0]),{},float,np.nan]
    for b in bad_value:
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

    with pytest.raises(ValueError):
        check_int(1,maximum_allowed=0)

    with pytest.raises(ValueError):
        check_int(1,maximum_allowed=1,maximum_inclusive=False)

    value = check_int(1,minimum_allowed=1,maximum_inclusive=True)
    assert value == 1


def test_column_to_bool():

    df = pd.DataFrame({"test":[True,False,
                               "True","False",
                               1,0,
                               "1","0",
                               "yes","no",
                               "T","F",
                               "Y","N"]})

    expected = np.array((1,0,1,0,1,0,1,0,1,0,1,0,1,0),dtype=bool)
    out = column_to_bool(df["test"],"test")
    assert np.array_equal(out,expected)

    df = pd.DataFrame({"test":[True,False,
                               "True","False",
                               1,0,
                               "1","0",
                               "yes","no",
                               "T","F",
                               "X","N"]})
    with pytest.raises(ValueError):
        out = column_to_bool(df["test"],"test")
