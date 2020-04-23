import pytest

import likelihood

import numpy as np


def test_init():

    f = likelihood.MLFitter()
    assert f.fit_type == "maximum likelihood"

def test_fit(binding_curve_test_data,fit_tolerance_fixture):
    """
    Test the ability to fit the test data in binding_curve_test_data.
    """

    f = likelihood.MLFitter()

    model = binding_curve_test_data["prewrapped_model"]
    guesses = binding_curve_test_data["guesses"]
    df = binding_curve_test_data["df"]
    input_params = np.array(binding_curve_test_data["input_params"])

    f.fit(model=model,guesses=guesses,y_obs=df.Y)

    # Make sure fit worked
    assert f.success

    # Make sure fit gave right answer
    assert np.allclose(f.estimate,
                       input_params,
                       rtol=fit_tolerance_fixture,
                       atol=fit_tolerance_fixture*input_params)

    # Make sure mean of sampled uncertainty gives right answer
    sampled = np.mean(f.samples,axis=0)
    assert np.allclose(f.estimate,
                       sampled,
                       rtol=fit_tolerance_fixture,
                       atol=fit_tolerance_fixture*input_params)

    # Make sure corner plot call works and generates a plot
    corner_fig = f.corner_plot()
    assert corner_fig is not None
