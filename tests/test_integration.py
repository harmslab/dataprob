
import pytest

import likelihood

import numpy as np


def test_integrated_ml_fit(binding_curve_test_data,fit_tolerance_fixture):

    df = binding_curve_test_data["df"]
    model_to_wrap = binding_curve_test_data["wrappable_model"]

    mw = likelihood.ModelWrapper(model_to_wrap)
    assert mw.df is None
    mw.df = df
    mw.K.bounds = [0,np.inf]

    f = likelihood.MLFitter()
    f.fit(mw,y_obs=df.Y,y_stdev=df.Y_stdev)
    assert f.bounds is not None
    assert f.bounds[0,0] == 0
    assert f.bounds[1,0] == np.inf

    assert f.success

    # Make sure fit gave right answer
    input_params = np.array(binding_curve_test_data["input_params"])
    assert np.allclose(f.estimate,
                       input_params,
                       rtol=fit_tolerance_fixture,
                       atol=fit_tolerance_fixture*input_params)

    f.fit_as_df
