
import pytest

from dataprob.model_wrapper.wrap_function import wrap_function
from dataprob.model_wrapper.model_wrapper import ModelWrapper
from dataprob.model_wrapper.vector_model_wrapper import VectorModelWrapper

import numpy as np
import pandas as pd

import os

def test_wrap_function(tmpdir):

    cwd = os.getcwd()
    os.chdir(tmpdir)

    # --------------------------------------------------------------
    # vector_first_arg is False

    def model_to_test_wrap(a,b=2,c=3,d="test",e=3): return a*b*c

    # basic test
    mw = wrap_function(some_function=model_to_test_wrap,
                       fit_parameters=None,
                       vector_first_arg=False)
    
    assert len(mw.param_df) == 3
    assert mw.param_df.loc["a","guess"] == 0
    assert mw.param_df.loc["b","guess"] == 2
    assert mw.param_df.loc["c","guess"] == 3

    assert issubclass(type(mw),ModelWrapper)

    # send in list of fit parameters
    mw = wrap_function(some_function=model_to_test_wrap,
                       fit_parameters=["a","b"],
                       vector_first_arg=False)
    
    assert len(mw.param_df) == 2
    assert mw.param_df.loc["a","guess"] == 0
    assert mw.param_df.loc["b","guess"] == 2

    assert issubclass(type(mw),ModelWrapper)

    # send in dict of fit parameters
    mw = wrap_function(some_function=model_to_test_wrap,
                       fit_parameters={"a":{"guess":20}},
                       vector_first_arg=False)
    
    assert len(mw.param_df) == 1
    assert mw.param_df.loc["a","guess"] == 20

    assert issubclass(type(mw),ModelWrapper)

    # send in dataframe of fit parameters
    df = pd.DataFrame({"name":["a"],
                       "guess":[20]})
    
    mw = wrap_function(some_function=model_to_test_wrap,
                       fit_parameters=df,
                       vector_first_arg=False)
    
    assert len(mw.param_df) == 1
    assert mw.param_df.loc["a","guess"] == 20

    assert issubclass(type(mw),ModelWrapper)

    # send in dataframe without name column
    df = pd.DataFrame({"not_name":["a"],
                       "guess":[20]})
    with pytest.raises(ValueError):
        mw = wrap_function(some_function=model_to_test_wrap,
                        fit_parameters=df,
                        vector_first_arg=False)


    # send in spreadsheet file of fit parameters
    df = pd.DataFrame({"name":["a"],
                       "guess":[20]})
    df.to_csv("dataframe.csv")
    
    mw = wrap_function(some_function=model_to_test_wrap,
                       fit_parameters="dataframe.csv",
                       vector_first_arg=False)
    
    assert len(mw.param_df) == 1
    assert mw.param_df.loc["a","guess"] == 20

    assert issubclass(type(mw),ModelWrapper)

    with pytest.raises(ValueError):
        mw = wrap_function(some_function=model_to_test_wrap,
                           fit_parameters=1,
                           vector_first_arg=False)

    # --------------------------------------------------------------
    # vector_first_arg is True

    def model_to_test_wrap(some_vector,q=3): return np.sum(some_vector)

    # basic test. will die because VectorModelWrapper requires non-None 
    # fit_parameters
    with pytest.raises(ValueError):
        mw = wrap_function(some_function=model_to_test_wrap,
                        fit_parameters=None,
                        vector_first_arg=True)
    
    # send in list of fit parameters
    mw = wrap_function(some_function=model_to_test_wrap,
                       fit_parameters=["a","b"],
                       vector_first_arg=True)
    
    assert len(mw.param_df) == 2
    assert mw.param_df.loc["a","guess"] == 0
    assert mw.param_df.loc["b","guess"] == 0

    assert issubclass(type(mw),VectorModelWrapper)

    # send in dict of fit parameters
    mw = wrap_function(some_function=model_to_test_wrap,
                       fit_parameters={"a":{"guess":20}},
                       vector_first_arg=True)
    
    assert len(mw.param_df) == 1
    assert mw.param_df.loc["a","guess"] == 20

    assert issubclass(type(mw),VectorModelWrapper)

    # send in dataframe of fit parameters
    df = pd.DataFrame({"name":["a"],
                       "guess":[20]})
    
    mw = wrap_function(some_function=model_to_test_wrap,
                       fit_parameters=df,
                       vector_first_arg=True)
    
    assert len(mw.param_df) == 1
    assert mw.param_df.loc["a","guess"] == 20

    assert issubclass(type(mw),VectorModelWrapper)

    # send in dataframe without name column
    df = pd.DataFrame({"not_name":["a"],
                       "guess":[20]})
    with pytest.raises(ValueError):
        mw = wrap_function(some_function=model_to_test_wrap,
                        fit_parameters=df,
                        vector_first_arg=True)


    # send in spreadsheet file of fit parameters
    df = pd.DataFrame({"name":["a"],
                       "guess":[20]})
    df.to_csv("dataframe.csv")
    
    mw = wrap_function(some_function=model_to_test_wrap,
                       fit_parameters="dataframe.csv",
                       vector_first_arg=True)
    
    assert len(mw.param_df) == 1
    assert mw.param_df.loc["a","guess"] == 20

    assert issubclass(type(mw),VectorModelWrapper)

    with pytest.raises(ValueError):
        mw = wrap_function(some_function=model_to_test_wrap,
                           fit_parameters=1,
                           vector_first_arg=True)

   # --------------------------------------------------------------
    # vector_first_arg is stupid

    def model_to_test_wrap(some_vector,q=3): return np.sum(some_vector)

    # basic test. will die because VectorModelWrapper requires non-None 
    # fit_parameters
    with pytest.raises(ValueError):
        mw = wrap_function(some_function=model_to_test_wrap,
                           fit_parameters=["a"],
                           vector_first_arg="stupid")

    os.chdir(cwd)