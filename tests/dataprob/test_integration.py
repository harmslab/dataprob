
import pytest

from dataprob.model_wrapper.model_wrapper import ModelWrapper

from dataprob.fitters.ml import MLFitter
from dataprob.fitters.bayesian.bayesian_sampler import BayesianSampler

import numpy as np


def test_integrated_ml_fit(binding_curve_test_data,
                           fit_tolerance_fixture):

    df = binding_curve_test_data["df"]
    model_to_wrap = binding_curve_test_data["wrappable_model"]

    mw = ModelWrapper(model_to_wrap)
    assert mw.df is None
    mw.df = df
    mw.param_df.loc["K","lower_bound"] = 0
    
    f = MLFitter()
    f.fit(mw,
          y_obs=df.Y,
          y_std=df.Y_stdev)
    assert f.success

    # Make sure fit gave right answer
    input_params = np.array(binding_curve_test_data["input_params"])
    assert np.allclose(f.fit_df["estimate"],
                       input_params,
                       rtol=fit_tolerance_fixture,
                       atol=fit_tolerance_fixture*input_params)

@pytest.mark.slow
def test_integrated_bayesian_fit(binding_curve_test_data,
                                 fit_tolerance_fixture):

    df = binding_curve_test_data["df"]
    model_to_wrap = binding_curve_test_data["wrappable_model"]

    mw = ModelWrapper(model_to_wrap)
    assert mw.df is None
    mw.df = df
    mw.param_df.loc["K","lower_bound"] = 0
    
    f = BayesianSampler()
    f.fit(mw,
          y_obs=df.Y,
          y_std=df.Y_stdev)
    assert f.success

    # Make sure fit gave right answer
    input_params = np.array(binding_curve_test_data["input_params"])
    assert np.allclose(f.fit_df["estimate"],
                       input_params,
                       rtol=fit_tolerance_fixture,
                       atol=fit_tolerance_fixture*input_params)
