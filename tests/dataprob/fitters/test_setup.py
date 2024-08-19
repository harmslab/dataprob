
import pytest

from dataprob.fitters.setup import setup

from dataprob.fitters.ml import MLFitter
from dataprob.fitters.bootstrap import BootstrapFitter
from dataprob.fitters.bayesian.bayesian_sampler import BayesianSampler

from dataprob.model_wrapper.model_wrapper import ModelWrapper
from dataprob.model_wrapper.vector_model_wrapper import VectorModelWrapper

import numpy as np

def test_setup():

    def test_fcn(a,b=2): return a*b

    with pytest.raises(ValueError):
        setup(some_function="not_a_function")

    # Test method passing (including default)
    f = setup(some_function=test_fcn)
    assert issubclass(type(f),MLFitter)

    f = setup(some_function=test_fcn,
              method="ml")
    assert issubclass(type(f),MLFitter)

    f = setup(some_function=test_fcn,
              method="bootstrap")
    assert issubclass(type(f),BootstrapFitter)

    f = setup(some_function=test_fcn,
              method="mcmc")
    assert issubclass(type(f),BayesianSampler)

    with pytest.raises(ValueError):
        f = setup(some_function=test_fcn,
                  method="not_a_method")

    # Test fit_parameters passing
    f = setup(some_function=test_fcn)
    assert issubclass(type(f),MLFitter)
    assert len(f.non_fit_kwargs) == 0

    f = setup(some_function=test_fcn,fit_parameters=["b"])
    assert issubclass(type(f),MLFitter)
    assert len(f.non_fit_kwargs) == 1
    assert f.non_fit_kwargs["a"] is None

    # Test non_fit_kwargs passing
    f = setup(some_function=test_fcn)
    assert issubclass(type(f),MLFitter)
    assert len(f.non_fit_kwargs) == 0

    f = setup(some_function=test_fcn,non_fit_kwargs={"b":2})
    assert issubclass(type(f),MLFitter)
    assert len(f.non_fit_kwargs) == 1
    assert f.non_fit_kwargs["b"] == 2
    
    # test vector_first_arg passing
    f = setup(some_function=test_fcn)
    assert issubclass(type(f),MLFitter)
    assert issubclass(type(f._model),ModelWrapper)

    # test vector_first_arg passing
    f = setup(some_function=test_fcn,
              vector_first_arg=False)
    assert issubclass(type(f),MLFitter)
    assert issubclass(type(f._model),ModelWrapper)

    # test vector_first_arg passing
    f = setup(some_function=test_fcn,
              fit_parameters=["x","y","z"],
              vector_first_arg=True)
    assert issubclass(type(f),MLFitter)
    assert issubclass(type(f._model),VectorModelWrapper)
    assert np.array_equal(f.param_df["name"],["x","y","z"])


    
