import pytest

import likelihood

import numpy as np

def model_to_test(K1=10,K2=20,extra_stuff="test"):

    return K1*K2


def test_init():

    mw = likelihood.ModelWrapper(model_to_test)

    params = list(mw.fit_parameters.keys())
    assert params[0] == "K1" and params[1] == "K2"

    args = list(mw.other_arguments.keys())
    assert args[0] == "extra_stuff"

    assert np.array_equal(mw.guesses,np.array((10,20)))
    assert np.array_equal(mw.bounds[0],np.array((-np.inf,-np.inf)))
    assert np.array_equal(mw.bounds[1],np.array((np.inf,np.inf)))
    assert mw.param_names[0] == "K1"
    assert mw.param_names[1] == "K2"


def test_fixing():

    # Wrap model
    mw = likelihood.ModelWrapper(model_to_test)
    assert mw.param_names[0] == "K1"
    assert mw.param_names[1] == "K2"

    # Fix one parameter
    mw.K1.fixed = True
    assert mw.param_names[0] == "K2"
    assert mw.guesses[0] == 20
    with pytest.raises(IndexError):
        mw.param_names[1]

    # Fix second parameter
    mw.K2.fixed = True
    with pytest.raises(IndexError):
        mw.param_names[0]

    # Unfix a parameter
    mw.K1.fixed = False
    assert mw.param_names[0] == "K1"
    assert mw.guesses[0] == 10
    with pytest.raises(IndexError):
        mw.param_names[1]

    # Try to fix a parameter that is not really a parameter
    with pytest.raises(AttributeError):
        mw.extra_stuff.fixed = True


def test_setting():

    # Wrap model
    mw = likelihood.ModelWrapper(model_to_test)
    assert mw.param_names[0] == "K1"
    assert mw.param_names[1] == "K2"

    # Fix one parameter
    mw.K1.fixed = True
    assert mw.param_names[0] == "K2"
    assert mw.guesses[0] == 20
    with pytest.raises(IndexError):
        mw.param_names[1]

    # Fix second parameter
    mw.K2.fixed = True
    with pytest.raises(IndexError):
        mw.param_names[0]

    # Unfix a parameter
    mw.K1.fixed = False
    assert mw.param_names[0] == "K1"
    assert mw.guesses[0] == 10
    with pytest.raises(IndexError):
        mw.param_names[1]

    # Try to fix a parameter that is not really a parameter
    with pytest.raises(AttributeError):
        mw.extra_stuff.fixed = True
