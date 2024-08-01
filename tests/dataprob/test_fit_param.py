import pytest

from dataprob.fit_param import _guess_setter_from_bounds
from dataprob.fit_param import FitParameter

import numpy as np

def test__guess_setter_from_bounds():

    from dataprob.fit_param import _INFINITY_PROXY
    
    guess = _guess_setter_from_bounds([-np.inf,np.inf])
    expected_guess = 0.0
    assert np.isclose(guess,expected_guess)

    guess = _guess_setter_from_bounds([1,np.inf])
    expected_guess = np.exp((np.log(1) + np.log(_INFINITY_PROXY))/2)
    assert np.isclose(guess,expected_guess)

    guess = _guess_setter_from_bounds([-np.inf,-1])
    expected_guess = -np.exp((np.log(1) + np.log(_INFINITY_PROXY))/2)
    assert np.isclose(guess,expected_guess)

    guess = _guess_setter_from_bounds([-1,np.inf])
    expected_guess = (-1 + _INFINITY_PROXY)/2
    assert np.isclose(guess,expected_guess)

    guess = _guess_setter_from_bounds([-np.inf,1])
    expected_guess = (-_INFINITY_PROXY + 1)/2
    assert np.isclose(guess,expected_guess)

    guess = _guess_setter_from_bounds([0,1])
    expected_guess = 0.5
    assert np.isclose(guess,expected_guess)
    
    guess = _guess_setter_from_bounds([-1,0])
    expected_guess = -0.5
    assert np.isclose(guess,expected_guess)
    


def test_init():

    with pytest.raises(TypeError):
        p = FitParameter()

    p = FitParameter(name="test")

    assert np.array_equal(p.bounds,np.array((-np.inf,np.inf)))
    assert p.guess == 0.0
    assert p.value == p.guess
    assert p.fixed == False
    assert np.array_equal(p.prior,[np.nan,np.nan],equal_nan=True)


def test_name_getter_setter():

    # Default (must be set via __init__)
    p = FitParameter(name="test")
    assert p.name == "test"

    # Set directly
    p.name = 5
    assert p.name == "5"

    # Set directly
    p.name = "junk"
    assert p.name == "junk"


def test_guess_getter_setter():

    # Default
    p = FitParameter(name="test")
    assert np.array_equal(p.bounds,np.array((-np.inf,np.inf)))
    assert p.guess == 0.0

    # Set via __init__
    p = FitParameter(name="test",guess=12)
    assert p.guess == 12

    # Set directly
    p = FitParameter(name="test")
    p.guess = 22
    assert p.guess == 22

    # --- bad value checks ---
    p = FitParameter(name="test")

    with pytest.raises(ValueError):
        p.guess = "test"

    with pytest.raises(ValueError):
        p.guess = [1.0]

    with pytest.raises(ValueError):
        p.guess = np.arange(10)

    # Set by prior
    p = FitParameter(name="test",prior=[10,1])
    assert np.array_equal(p.bounds,np.array((-np.inf,np.inf)))
    assert p.guess == 10.0

    # Set by bounds (geometric mean)
    p = FitParameter(name="test",bounds=[1,3])
    assert np.array_equal(p.bounds,np.array((1,3)))
    predicted_guess = np.exp((np.log(1) + np.log(3))/2)
    assert np.isclose(p.guess,predicted_guess)

    # Set by bounds (arithmetic mean)
    p = FitParameter(name="test",bounds=[-2,3])
    assert np.array_equal(p.bounds,np.array((-2,3)))
    predicted_guess = (-2 + 3)/2
    assert np.isclose(p.guess,predicted_guess)

    # Prior overrides bounds
    p = FitParameter(name="test",prior=[10,1],bounds=[0,11])
    assert np.array_equal(p.bounds,np.array((0,11)))
    assert p.guess == 10.0

    # Infinite left bound
    INFINITY_PROXY = 1e9
    p = FitParameter(name="test",bounds=[-np.inf,11])
    assert np.array_equal(p.bounds,np.array((-np.inf,11)))
    predicted_guess = (-INFINITY_PROXY + 3)/2
    assert np.isclose(p.guess,predicted_guess)

    # Infinite right bound
    p = FitParameter(name="test",bounds=[1,np.inf])
    assert np.array_equal(p.bounds,np.array((1,np.inf)))
    predicted_guess = np.exp((np.log(1) + np.log(INFINITY_PROXY))/2)
    assert np.isclose(p.guess,predicted_guess)

    # --- bound check ---
    with pytest.raises(ValueError):
        p = FitParameter(name="test")
        p.bounds = [-10,10]
        p.guess = -20

    with pytest.raises(ValueError):
        p = FitParameter(name="test")
        p.bounds = [-10,10]
        p.guess = 20

def test_fixed_setter_getter():

    p = FitParameter(name="test")
    assert p.fixed is False
    p.fixed = True
    assert p.fixed is True
    p.fixed = False
    assert p.fixed is False

    # send in a wacky value to make sure bool check is being run
    p = FitParameter(name="test")
    with pytest.raises(ValueError):
        p.fixed = "not_a_good_bool"

def test_bounds_getter_setter():

    # Default
    p = FitParameter(name="test")
    assert np.array_equal(p.bounds,np.array((-np.inf,np.inf)))

    # Set via __init__
    bounds = [1,2]
    p = FitParameter(name="test",bounds=bounds)
    assert np.array_equal(p.bounds,np.array(bounds))

    # Set directly
    bounds = [1,2]
    p.bounds = bounds
    assert np.array_equal(p.bounds,np.array(bounds))

    # --- bad value checks ---
    with pytest.raises(ValueError):
        p.bounds = "test"

    with pytest.raises(ValueError):
        p.bounds = "te"

    with pytest.raises(ValueError):
        p.bounds = 1.0

    with pytest.raises(ValueError):
        p.bounds = [1.0]

    with pytest.raises(ValueError):
        p.bounds = [1.0,1.0]

    with pytest.raises(ValueError):
        p.bounds = [1.0,-1.0]

    with pytest.raises(ValueError):
        p.bounds = ["a","b"]

    # Identical bounds
    p = FitParameter(name="test")
    with pytest.raises(ValueError):
        p.bounds = np.zeros(2,dtype=float)

    p = FitParameter(name="test")
    with pytest.raises(ValueError):
        p.bounds = 10*np.ones(2,dtype=float)

    # Numerically close to zero...
    input_bounds = np.zeros(2,dtype=float)
    resolution = np.finfo(input_bounds.dtype).resolution

    p = FitParameter(name="test")
    input_bounds[1] = resolution/10
    with pytest.raises(ValueError):
        p.bounds = input_bounds

    # But now far enough away that it is not zero
    input_bounds[1] = resolution*10
    p = FitParameter(name="test")
    p.bounds = input_bounds
    assert np.array_equal(p.bounds,input_bounds)

    # Shift guess down to within bounds
    p = FitParameter(name="test",guess=2)
    p.bounds = [1,3]
    with pytest.raises(ValueError):
        p.guess = 10

    p.guess = 2
    assert np.array_equal(p.bounds,[1,3])
    assert p.guess == 2
    
    with pytest.warns():
        p.bounds = [1,1.5]
    assert np.allclose(p.bounds,[1,1.5])
    assert np.isclose(p.guess,1.5)

    # shfit guess up to within bounds
    p = FitParameter(name="test",guess=2)
    p.bounds = [1,3]
    with pytest.warns():
        p.bounds = [2.5,3]
    assert np.allclose(p.bounds,[2.5,3.0])
    assert np.isclose(p.guess,2.5)


def test_prior_setter_getter():

    # Default
    p = FitParameter(name="test")
    assert np.array_equal(p.prior,[np.nan,np.nan],equal_nan=True)

    # Set via __init__
    prior = [1,2]
    p = FitParameter(name="test",prior=prior)
    assert np.array_equal(p.prior,np.array(prior))

    # Set directly
    prior = [1,2]
    p.prior = prior
    assert np.array_equal(p.prior,np.array(prior))
    
    # --- bad value checks ---
    with pytest.raises(ValueError):
        p.prior = "test"

    with pytest.raises(ValueError):
        p.prior = "te"

    with pytest.raises(ValueError):
        p.prior = 1.0

    with pytest.raises(ValueError):
        p.prior = [1.0]

    # can't be negative second number
    with pytest.raises(ValueError):
        p.bounds = [1.0,-1.0]

    with pytest.raises(ValueError):
        p.bounds = ["a","b"]

    # make sure guess gets set from prior if defined
    p = FitParameter(name="test")
    assert p.guess == 0
    p.prior = [1,2]
    assert np.array_equal(p.prior,[1,2])
    p.guess = None
    assert p.guess == 1


def test_value_getter():
    p = FitParameter(name="test")
    assert p.value == 0.0 # guess
    p._value = 10
    assert p.value == 10

def test_stdev_getter():
    p = FitParameter(name="test")
    assert p.stdev is None
    p._stdev = 10
    assert p.stdev == 10

def test_stdev_getter():
    p = FitParameter(name="test")
    assert p.ninetyfive is None
    p._ninetyfive = [1,2]
    assert np.array_equal(p.ninetyfive,[1,2])

def test_is_fit_result_getter():
    p = FitParameter(name="test")
    assert p.is_fit_result is False
    p._is_fit_result = True
    assert p.is_fit_result is True


def test_load_clear_fit_results(fitter_object):

    # --- Make sure we can load fit result into parameter ---
    p = FitParameter(name="test")
    assert p.value == p.guess
    assert p.stdev is None
    assert p.ninetyfive is None
    assert np.array_equal(p.prior,[np.nan,np.nan],equal_nan=True)
    assert not p.is_fit_result

    p.load_fit_result(fitter_object,0)
    assert p.value == fitter_object.estimate[0]
    assert p.stdev == fitter_object.stdev[0]
    assert np.array_equal(p.ninetyfive[0],fitter_object.ninetyfive[0,0])
    assert np.array_equal(p.ninetyfive[1],fitter_object.ninetyfive[1,0])
    assert p.is_fit_result

    # --- Make sure setting guess wipes out fit ---
    p.guess = 22
    assert p.guess == 22
    assert p.value == p.guess
    assert p.stdev is None
    assert p.ninetyfive is None
    assert not p.is_fit_result

    # --- Make sure setting bounds wipes out fit ---

    p = FitParameter(name="test")
    assert p.value == p.guess
    assert p.stdev is None
    assert p.ninetyfive is None
    assert np.array_equal(p.bounds,np.array((-np.inf,np.inf)))
    assert np.array_equal(p.prior,[np.nan,np.nan],equal_nan=True)
    assert not p.is_fit_result

    p.load_fit_result(fitter_object,0)
    assert p.value == fitter_object.estimate[0]
    assert p.stdev == fitter_object.stdev[0]
    assert np.array_equal(p.ninetyfive[0],fitter_object.ninetyfive[0,0])
    assert np.array_equal(p.ninetyfive[1],fitter_object.ninetyfive[1,0])
    assert p.is_fit_result

    p.bounds = [-100,100]
    assert np.array_equal(p.bounds,np.array((-100,100)))
    assert np.array_equal(p.prior,[np.nan,np.nan],equal_nan=True)
    assert p.value == p.guess
    assert p.stdev is None
    assert p.ninetyfive is None
    assert not p.is_fit_result

    # --- Make sure setting prior wipes out fit ---

    p = FitParameter(name="test")
    assert p.value == p.guess
    assert p.stdev is None
    assert p.ninetyfive is None
    assert np.array_equal(p.bounds,np.array((-np.inf,np.inf)))
    assert np.array_equal(p.prior,[np.nan,np.nan],equal_nan=True)
    assert not p.is_fit_result

    p.load_fit_result(fitter_object,0)
    assert p.value == fitter_object.estimate[0]
    assert p.stdev == fitter_object.stdev[0]
    assert np.array_equal(p.ninetyfive[0],fitter_object.ninetyfive[0,0])
    assert np.array_equal(p.ninetyfive[1],fitter_object.ninetyfive[1,0])
    assert p.is_fit_result

    p.prior = [2,1]
    assert np.array_equal(p.bounds,np.array((-np.inf,np.inf)))
    assert np.array_equal(p.prior,[2,1])
    assert p.value == p.guess
    assert p.stdev is None
    assert p.ninetyfive is None
    assert not p.is_fit_result

    # --- Make sure setting fixed wipes out fit ---

    p = FitParameter(name="test")
    assert p.value == p.guess
    assert p.stdev is None
    assert p.ninetyfive is None
    assert p.fixed == False
    assert not p.is_fit_result

    p.load_fit_result(fitter_object,0)
    assert p.value == fitter_object.estimate[0]
    assert p.stdev == fitter_object.stdev[0]
    assert np.array_equal(p.ninetyfive[0],fitter_object.ninetyfive[0,0])
    assert np.array_equal(p.ninetyfive[1],fitter_object.ninetyfive[1,0])
    assert p.is_fit_result

    p.fixed = True
    assert p.fixed == True
    assert p.value == p.guess
    assert p.stdev is None
    assert p.ninetyfive is None
    assert not p.is_fit_result




def xtest_interaction_bounds_guesses():

    # --- default guess depends on bounds; try different bounds scenarios ---

    # Two positive bounds
    bounds = [10,20]
    p = FitParameter(name="test",bounds=bounds)
    assert np.array_equal(p.bounds,np.array(bounds))
    assert np.isclose(p.guess,np.exp(np.mean(np.log(bounds))))

    # Two negative bounds
    bounds = [-20,-10]
    p = FitParameter(name="test",bounds=bounds)
    assert np.array_equal(p.bounds,np.array(bounds))
    assert np.isclose(p.guess,-np.exp(np.mean(np.log(np.abs(bounds)))))

    # One negative, one positive
    bounds = [-10,10]
    p = FitParameter(name="test",bounds=bounds)
    assert np.array_equal(p.bounds,np.array(bounds))
    assert np.isclose(p.guess,np.sum(bounds)/2)

    # negative infinity, positive real
    bounds = [-np.inf,10]
    internal_bounds = [-dataprob.fit_param._INFINITY_PROXY,bounds[1]]
    expected = np.sum(internal_bounds)/2
    p = FitParameter(name="test",bounds=bounds)
    assert np.array_equal(p.bounds,np.array(bounds))
    assert np.isclose(p.guess,expected)

    # negative real, positive infinity
    bounds = [-10,np.inf]
    p = FitParameter(name="test",bounds=bounds)
    internal_bounds = [bounds[0],dataprob.fit_param._INFINITY_PROXY]
    expected = np.sum(internal_bounds)/2
    assert np.array_equal(p.bounds,np.array(bounds))
    assert np.isclose(p.guess,expected)

    # negative infinity, positive infinity
    bounds = [-np.inf,np.inf]
    p = FitParameter(name="test",bounds=bounds)
    assert np.array_equal(p.bounds,np.array(bounds))
    assert p.guess == 0.0

    # --- Update bounds such that guess is outside the new bounds ---

    bounds = [-10,10]
    p = FitParameter(name="test",bounds=bounds,guess=-5)
    assert np.array_equal(p.bounds,np.array(bounds))
    assert p.guess == -5.0

    new_bounds = [0,10]
    with pytest.warns(UserWarning):
        p.bounds = new_bounds
    assert np.array_equal(p.bounds,np.array(new_bounds))
    assert p.guess == new_bounds[0]

    bounds = [-10,10]
    p = FitParameter(name="test",bounds=bounds,guess=5)
    assert np.array_equal(p.bounds,np.array(bounds))
    assert p.guess == 5.0

    new_bounds = [-10,0]
    with pytest.warns(UserWarning):
        p.bounds = new_bounds
    assert np.array_equal(p.bounds,np.array(new_bounds))
    assert p.guess == new_bounds[1]

    # --- Upddate bounds such tthat guess remains in the new bounds ---
    bounds = [-10,10]
    p = FitParameter(name="test",bounds=bounds,guess=5)
    assert np.array_equal(p.bounds,np.array(bounds))
    assert p.guess == 5.0

    new_bounds = [0,10]
    p.bounds = new_bounds
    assert np.array_equal(p.bounds,np.array(new_bounds))
    assert p.guess == 5.0
