
import pytest

from dataprob.model_wrapper.model_wrapper import ModelWrapper

from dataprob.fitters.ml import MLFitter
from dataprob.fitters.bayesian.bayesian_sampler import BayesianSampler
from dataprob.fitters.bootstrap import BootstrapFitter

from dataprob.plot import plot_corner
from dataprob.plot import plot_summary

import numpy as np
import matplotlib
from matplotlib import pyplot as plt


def _integrated_binding_curve_fit(fitter,
                                  binding_curve_test_data,
                                  fit_tolerance_fixture):

    print("testing binding curve fit for",fitter)

    df = binding_curve_test_data["df"]
    model_to_wrap = binding_curve_test_data["wrappable_model"]

    f = fitter(some_function=model_to_wrap,
               non_fit_kwargs={"df":df})
    f.param_df.loc["K","lower_bound"] = 0
    f.fit(y_obs=df.y_obs,
          y_std=df.y_std)
    assert f.success

    # Make sure fit gave right answer
    input_params = np.array(binding_curve_test_data["input_params"])
    assert np.allclose(f.fit_df["estimate"],
                    input_params,
                    rtol=fit_tolerance_fixture,
                    atol=fit_tolerance_fixture*input_params)
    
    # Plot
    fig = plot_summary(f)
    assert issubclass(type(fig),matplotlib.figure.Figure)
    plt.close(fig)

    fig = plot_corner(f)
    assert issubclass(type(fig),matplotlib.figure.Figure)
    plt.close(fig)
    

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





