
import pytest

from dataprob.model_wrapper.model_wrapper import ModelWrapper

from dataprob.fitters.ml import MLFitter
from dataprob.fitters.bayesian.bayesian_sampler import BayesianSampler
from dataprob.fitters.bootstrap import BootstrapFitter

import numpy as np


def _integrated_binding_curve_fit(fitter,
                                  binding_curve_test_data,
                                  fit_tolerance_fixture):

    print("testing binding curve fit for",fitter)

    df = binding_curve_test_data["df"]
    model_to_wrap = binding_curve_test_data["wrappable_model"]

    f = fitter(some_function=model_to_wrap,
               non_fit_kwargs={"df":df})
    f.param_df.loc["K","lower_bound"] = 0
    f.fit(y_obs=df.Y,
          y_std=df.Y_stdev)
    assert f.success

    # Make sure fit gave right answer
    input_params = np.array(binding_curve_test_data["input_params"])
    assert np.allclose(f.fit_df["estimate"],
                    input_params,
                    rtol=fit_tolerance_fixture,
                    atol=fit_tolerance_fixture*input_params)
    

def test_ml_binding_curve(binding_curve_test_data,
                          fit_tolerance_fixture):
    
    _integrated_binding_curve_fit(fitter=MLFitter,
                                  binding_curve_test_data=binding_curve_test_data,
                                  fit_tolerance_fixture=fit_tolerance_fixture)

@pytest.mark.slow
def test_bayesian_binding_curve(binding_curve_test_data,
                                fit_tolerance_fixture):
    
    _integrated_binding_curve_fit(fitter=BayesianSampler,
                                  binding_curve_test_data=binding_curve_test_data,
                                  fit_tolerance_fixture=fit_tolerance_fixture)

@pytest.mark.slow
def test_bootstrap_binding_curve(binding_curve_test_data,
                                 fit_tolerance_fixture):
    
    _integrated_binding_curve_fit(fitter=BootstrapFitter,
                                  binding_curve_test_data=binding_curve_test_data,
                                  fit_tolerance_fixture=fit_tolerance_fixture)





