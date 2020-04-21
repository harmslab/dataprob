
import pytest

import likelihood
import numpy as np

import inspect


# ---------------------------------------------------------------------------- #
# Test __init__
# ---------------------------------------------------------------------------- #

def test_init():
    """
    Test model initialization.
    """

    f = likelihood.base.Fitter()
    assert f.fit_type == ""

# ---------------------------------------------------------------------------- #
# Test setters, getters, and internal sanity checks
# ---------------------------------------------------------------------------- #

def test_model_setter_getter(binding_curve_test_data):
    """
    Test the model setter.
    """

    f = likelihood.base.Fitter()

    with pytest.raises(ValueError):
        f.model = "a"
    with pytest.raises(ValueError):
        def dummy(): pass
        f.model = dummy
    f.model = binding_curve_test_data["prewrapped_model"]
    assert f.model is not None
    assert f.model == binding_curve_test_data["prewrapped_model"]

def test_guesses_setter_getter(binding_curve_test_data):
    """
    Test the guesses setter.
    """

    f = likelihood.base.Fitter()

    with pytest.raises(ValueError):
        f.guesses = "a"
    with pytest.raises(ValueError):
        def dummy(): pass
        f.guesses = dummy
    f.guesses = binding_curve_test_data["guesses"]
    assert f.guesses is not None
    assert np.array_equal(f.guesses,binding_curve_test_data["guesses"])

def test_bounds_setter_getter(binding_curve_test_data):
    """
    Test the bounds setter.
    """

    f = likelihood.base.Fitter()

    with pytest.raises(ValueError):
        f.bounds = "a"
    with pytest.raises(ValueError):
        def dummy(): pass
        f.bounds = dummy

    bnds = [[-np.inf for _ in range(len(binding_curve_test_data["guesses"]))],
            [ np.inf for _ in range(len(binding_curve_test_data["guesses"]))]]
    bnds = np.array(bnds)

    f.bounds = bnds
    assert f.bounds is not None
    assert np.array_equal(f.bounds,bnds)

def test_param_names_setter_getter(binding_curve_test_data):
    """
    Test the param_names setter.
    """

    f = likelihood.base.Fitter()

    param_names = ["p{}".format(i)
                   for i in range(len(binding_curve_test_data["guesses"]))]
    f.param_names = param_names
    assert f.param_names is not None
    assert np.array_equal(f.param_names,param_names)


def test_param_mismatch_check(binding_curve_test_data):
    """
    Test the check for mismatches in the number of parameters in guesses,
    bounds, and param_names.
    """

    f = likelihood.base.Fitter()

    f.guesses = binding_curve_test_data["guesses"]
    with pytest.raises(ValueError):
        f.param_names = ["p{}".format(i)
                         for i in range(len(binding_curve_test_data["guesses"])-1)]
    f.param_names = ["p{}".format(i)
                     for i in range(len(binding_curve_test_data["guesses"]))]

    with pytest.raises(ValueError):
        bnds = [[-np.inf for _ in range(len(binding_curve_test_data["guesses"])-1)],
                [ np.inf for _ in range(len(binding_curve_test_data["guesses"])-1)]]
        bnds = np.array(bnds)
        f.bounds = bnds

    bnds = [[-np.inf for _ in range(len(binding_curve_test_data["guesses"]))],
            [ np.inf for _ in range(len(binding_curve_test_data["guesses"]))]]
    bnds = np.array(bnds)
    f.bounds = bnds


def test_y_obs_setter_getter(binding_curve_test_data):
    """
    Test the y_obs setter.
    """

    f = likelihood.base.Fitter()

    with pytest.raises(ValueError):
        f.y_obs = "a"
    with pytest.raises(ValueError):
        def dummy(): pass
        f.y_obs = dummy
    f.y_obs = binding_curve_test_data["df"].Y
    assert f.y_obs is not None
    assert np.array_equal(f.y_obs,binding_curve_test_data["df"].Y)

def test_y_stdev_setter_getter(binding_curve_test_data):
    """
    Test the y_stdev setter.
    """

    f = likelihood.base.Fitter()

    with pytest.raises(ValueError):
        f.y_stdev = "a"
    with pytest.raises(ValueError):
        def dummy(): pass
        f.y_stdev = dummy
    f.y_stdev = binding_curve_test_data["df"].Y_stdev
    assert f.y_stdev is not None
    assert np.array_equal(f.y_stdev,binding_curve_test_data["df"].Y_stdev)

def test_obs_mismatch_check(binding_curve_test_data):
    """
    Test the check for mismatches in the number of observations in y_obs
    and y_stdev.
    """

    f = likelihood.base.Fitter()

    f.y_obs = binding_curve_test_data["df"].Y
    with pytest.raises(ValueError):
        f.y_stdev = binding_curve_test_data["df"].Y_stdev[:-1]
    f.y_stdev = binding_curve_test_data["df"].Y_stdev

    f = likelihood.base.Fitter()

    f.y_stdev = binding_curve_test_data["df"].Y_stdev
    with pytest.raises(ValueError):
        f.y_obs = binding_curve_test_data["df"].Y[:-1]
    f.y_obs = binding_curve_test_data["df"].Y


def test_fit_completeness_sanity_checking(binding_curve_test_data):

    f = likelihood.base.Fitter()

    # This should not work because we have not specified a model, guesses,
    # or y_obs yet
    with pytest.raises(likelihood.LikelihoodError):
        f.fit()

    f.model = binding_curve_test_data["prewrapped_model"]

    # This should not work because we have not specified guesses or y_obs
    # yet.
    with pytest.raises(likelihood.LikelihoodError):
        f.fit()

    f.guesses = binding_curve_test_data["guesses"]

    # This should not work because we have not specified y_obs yet
    with pytest.raises(likelihood.LikelihoodError):
        f.fit()

    f.y_obs = binding_curve_test_data["df"].Y

    # Should now work because we've set everything essential (model, gueses,
    # and y_obs)
    f.fit()


def test_model_wrapper_interface():

    assert False

# ---------------------------------------------------------------------------- #
# Test residuals and the like
# ---------------------------------------------------------------------------- #

def test_unweighted_residuals(binding_curve_test_data):
    """
    Test unweighted residuals call against "manual" code used to generate
    test data.
    """

    f = likelihood.base.Fitter()

    input_params = binding_curve_test_data["input_params"]

    # Should fail, haven't loaded a model or y_obs yet
    with pytest.raises(likelihood.LikelihoodError):
        f.unweighted_residuals(input_params)

    f.model = binding_curve_test_data["prewrapped_model"]

    # Should fail, haven't loaded y_obs yet
    with pytest.raises(likelihood.LikelihoodError):
        f.unweighted_residuals(input_params)

    df = binding_curve_test_data["df"]
    f.y_obs = df.Y

    r = f.unweighted_residuals(input_params)

    assert np.allclose(r,df.residual)


def test_weighted_residuals(binding_curve_test_data):
    """
    Test weighted residuals call against "manual" code used to generate
    test data.
    """

    f = likelihood.base.Fitter()

    input_params = binding_curve_test_data["input_params"]

    # Should fail, haven't loaded a model, y_obs or y_stdev yet
    with pytest.raises(likelihood.LikelihoodError):
        f.weighted_residuals(input_params)

    f.model = binding_curve_test_data["prewrapped_model"]

    # Should fail, haven't loaded y_obs or y_stdev yet
    with pytest.raises(likelihood.LikelihoodError):
        f.weighted_residuals(input_params)

    df = binding_curve_test_data["df"]
    f.y_obs = df.Y

    # Should fail, haven't loaded y_stdev yet
    with pytest.raises(likelihood.LikelihoodError):
        f.weighted_residuals(input_params)

    f.y_stdev = df.Y_stdev
    r = f.weighted_residuals(input_params)

    assert np.allclose(r,df.weighted_residual)


def test_ln_like(binding_curve_test_data):
    """
    Test log likelihood call against "manual" code used to generate
    test data.
    """

    f = likelihood.base.Fitter()

    input_params = binding_curve_test_data["input_params"]

    # Should fail, haven't loaded a model, y_obs or y_stdev yet
    with pytest.raises(likelihood.LikelihoodError):
        f.ln_like(input_params)

    f.model = binding_curve_test_data["prewrapped_model"]

    # Should fail, haven't loaded y_obs or y_stdev yet
    with pytest.raises(likelihood.LikelihoodError):
        f.ln_like(input_params)

    df = binding_curve_test_data["df"]
    f.y_obs = df.Y

    # Should fail, haven't loaded y_stdev yet
    with pytest.raises(likelihood.LikelihoodError):
        f.ln_like(input_params)

    f.y_stdev = df.Y_stdev
    L = f.ln_like(input_params)

    assert np.allclose(L,binding_curve_test_data["ln_like"])

def test_num_params():

    f = likelihood.base.Fitter()
    assert f.num_params is None

    f.guesses = np.array([1,2])
    assert f.num_params == 2

    with pytest.raises(ValueError):
        f.guesses = np.array([7,8,9,10])

    f = likelihood.base.Fitter()
    f.guesses = np.array([])
    assert f.num_params == 0

def test_num_obs():

    f = likelihood.base.Fitter()
    assert f.num_obs is None

    f.y_obs = np.arange(10)
    assert f.num_obs == 10

    with pytest.raises(ValueError):
        f.y_obs = np.arange(2)

    f = likelihood.base.Fitter()
    f.y_obs = np.array([])
    assert f.num_obs == 0


def test_base_properties():

    f = likelihood.base.Fitter()

    assert f.estimate is None
    assert f.stdev is None
    assert f.ninetyfive is None
    assert f.fit_result is None
    assert f.success is None
    assert f.fit_info is None
    assert f.samples is None

def test_base_functions():

    f = likelihood.base.Fitter()
    assert f.corner_plot() is None
