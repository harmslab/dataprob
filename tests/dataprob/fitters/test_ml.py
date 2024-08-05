import pytest

import dataprob

import numpy as np
import pandas as pd

def test_init():

    f = dataprob.MLFitter()
    assert f.fit_type == "maximum likelihood"

def test_fit(binding_curve_test_data,fit_tolerance_fixture):
    """
    Test the ability to fit the test data in binding_curve_test_data.
    """

    # Do fit using a generic model and then creating and using a ModelWrapper
    # around wrappable_model

    for model_key in ["generic_model","wrappable_model"]:

        f = dataprob.MLFitter()
        model = binding_curve_test_data[model_key]
        guesses = binding_curve_test_data["guesses"]
        df = binding_curve_test_data["df"]
        input_params = np.array(binding_curve_test_data["input_params"])

        if model_key == "wrappable_model":
            model = dataprob.ModelWrapper(model)
            model.df = df
            model.K.bounds = [0,10]
        else:
            f.bounds = [[0],[10]]

        f.fit(model=model,guesses=guesses,y_obs=df.Y,y_stdev=df.Y_stdev)

        # Assert that we succesfully passed in bounds
        assert np.allclose(f.bounds,np.array([[0],[10]]))

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

        # Make sure data frame that comes out is correct
        df = f.fit_df

        assert isinstance(df,pd.DataFrame)
        assert np.allclose(df["estimate"].iloc[:],
                           input_params,
                           rtol=fit_tolerance_fixture,
                           atol=fit_tolerance_fixture*input_params)
        assert np.array_equal(df["param"],f.names)
        assert np.array_equal(df["estimate"],f.estimate)
        assert np.array_equal(df["stdev"],f.stdev)
        assert np.array_equal(df["low_95"],f.ninetyfive[0,:])
        assert np.array_equal(df["high_95"],f.ninetyfive[1,:])
        assert np.array_equal(df["guess"],f.guesses)
        assert np.array_equal(df["lower_bound"],f.bounds[0,:])
        assert np.array_equal(df["upper_bound"],f.bounds[1,:])
