import pytest

import likelihood

import numpy as np

def test_init():

    with pytest.raises(TypeError):
        p = likelihood.FitParameter()

    p = likelihood.FitParameter(name="test")

    assert np.array_equal(p.bounds,np.array((-np.inf,np.inf)))
    assert p.guess == 1.0
    assert p.value == p.guess
    assert p.fixed == False


def test_name_getter_setter():

    # Default (must be set via __init__)
    p = likelihood.FitParameter(name="test")
    assert p.name == "test"

    # Set directly
    p.name = 5
    assert p.name == "5"

    # Set directly
    p.name = "junk"
    assert p.name == "junk"


def test_guess_getter_setter():

    # Default
    p = likelihood.FitParameter(name="test")
    assert np.array_equal(p.bounds,np.array((-np.inf,np.inf)))
    assert p.guess == 1.0

    # Set via __init__
    p = likelihood.FitParameter(name="test",guess=12)
    assert p.guess == 12

    # Set directly
    p = likelihood.FitParameter(name="test")
    p.guess = 22
    assert p.guess == 22

    # --- bad value checks ---
    p = likelihood.FitParameter(name="test")

    with pytest.raises(ValueError):
        p.guess = "test"

    with pytest.raises(ValueError):
        p.guess = [1.0]

    with pytest.raises(ValueError):
        p.guess = np.arange(10)

    # --- bound check ---

    with pytest.raises(ValueError):
        p = likelihood.FitParameter(name="test")
        p.bounds = [-10,10]
        p.guess = -20

    with pytest.raises(ValueError):
        p = likelihood.FitParameter(name="test")
        p.bounds = [-10,10]
        p.guess = 20

def test_bounds_getter_setter():

    # Default
    p = likelihood.FitParameter(name="test")
    assert np.array_equal(p.bounds,np.array((-np.inf,np.inf)))

    # Set via __init__
    bounds = [1,2]
    p = likelihood.FitParameter(name="test",bounds=bounds)
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

def test_interaction_bounds_guesses():

    # --- default guess depends on bounds; try different bounds scenarios ---

    # Two positive bounds
    bounds = [10,20]
    p = likelihood.FitParameter(name="test",bounds=bounds)
    assert np.array_equal(p.bounds,np.array(bounds))
    assert np.isclose(p.guess,np.exp(np.mean(np.log(bounds))))

    # Two negative bounds
    bounds = [-20,-10]
    p = likelihood.FitParameter(name="test",bounds=bounds)
    assert np.array_equal(p.bounds,np.array(bounds))
    assert np.isclose(p.guess,-np.exp(np.mean(np.log(np.abs(bounds)))))

    # One negative, one positive
    bounds = [-10,10]
    p = likelihood.FitParameter(name="test",bounds=bounds)
    assert np.array_equal(p.bounds,np.array(bounds))
    assert np.isclose(p.guess,np.sum(bounds)/2)

    # negative infinity, positive real
    bounds = [-np.inf,10]
    internal_bounds = [-likelihood.fit_param._INFINITY_PROXY,bounds[1]]
    expected = np.sum(internal_bounds)/2
    p = likelihood.FitParameter(name="test",bounds=bounds)
    assert np.array_equal(p.bounds,np.array(bounds))
    assert np.isclose(p.guess,expected)

    # negative real, positive infinity
    bounds = [-10,np.inf]
    p = likelihood.FitParameter(name="test",bounds=bounds)
    internal_bounds = [bounds[0],likelihood.fit_param._INFINITY_PROXY]
    expected = np.sum(internal_bounds)/2
    assert np.array_equal(p.bounds,np.array(bounds))
    assert np.isclose(p.guess,expected)

    # negative infinity, positive infinity
    bounds = [-np.inf,np.inf]
    p = likelihood.FitParameter(name="test",bounds=bounds)
    assert np.array_equal(p.bounds,np.array(bounds))
    assert p.guess == 1.0

    # --- Update bounds such that guess is outside the new bounds ---

    bounds = [-10,10]
    p = likelihood.FitParameter(name="test",bounds=bounds,guess=-5)
    assert np.array_equal(p.bounds,np.array(bounds))
    assert p.guess == -5.0

    new_bounds = [0,10]
    with pytest.warns(UserWarning):
        p.bounds = new_bounds
    assert np.array_equal(p.bounds,np.array(new_bounds))
    assert p.guess == new_bounds[0]

    bounds = [-10,10]
    p = likelihood.FitParameter(name="test",bounds=bounds,guess=5)
    assert np.array_equal(p.bounds,np.array(bounds))
    assert p.guess == 5.0

    new_bounds = [-10,0]
    with pytest.warns(UserWarning):
        p.bounds = new_bounds
    assert np.array_equal(p.bounds,np.array(new_bounds))
    assert p.guess == new_bounds[1]

    # --- Upddate bounds such tthat guess remains in the new bounds ---
    bounds = [-10,10]
    p = likelihood.FitParameter(name="test",bounds=bounds,guess=5)
    assert np.array_equal(p.bounds,np.array(bounds))
    assert p.guess == 5.0

    new_bounds = [0,10]
    p.bounds = new_bounds
    assert np.array_equal(p.bounds,np.array(new_bounds))
    assert p.guess == 5.0



def test_load_clear_fit_results(fitter_object):

    # --- Make sure we can load fit result into parameter ---
    p = likelihood.FitParameter(name="test")
    assert p.value == p.guess
    assert p.stdev is None
    assert p.ninetyfive is None
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

    p = likelihood.FitParameter(name="test")
    assert p.value == p.guess
    assert p.stdev is None
    assert p.ninetyfive is None
    assert np.array_equal(p.bounds,np.array((-np.inf,np.inf)))
    assert not p.is_fit_result

    p.load_fit_result(fitter_object,0)
    assert p.value == fitter_object.estimate[0]
    assert p.stdev == fitter_object.stdev[0]
    assert np.array_equal(p.ninetyfive[0],fitter_object.ninetyfive[0,0])
    assert np.array_equal(p.ninetyfive[1],fitter_object.ninetyfive[1,0])
    assert p.is_fit_result

    p.bounds = [-100,100]
    assert np.array_equal(p.bounds,np.array((-100,100)))
    assert p.value == p.guess
    assert p.stdev is None
    assert p.ninetyfive is None
    assert not p.is_fit_result


    # --- Make sure setting fixed wipes out fit ---

    p = likelihood.FitParameter(name="test")
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
