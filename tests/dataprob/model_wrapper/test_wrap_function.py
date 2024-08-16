
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
                       non_fit_kwargs=None,
                       vector_first_arg=False)
    
    assert len(mw.param_df) == 3
    assert mw.param_df.loc["a","guess"] == 0
    assert mw.param_df.loc["b","guess"] == 2
    assert mw.param_df.loc["c","guess"] == 3

    assert issubclass(type(mw),ModelWrapper)

    # send in list of fit parameters
    mw = wrap_function(some_function=model_to_test_wrap,
                       fit_parameters=["a","b"],
                       non_fit_kwargs=None,
                       vector_first_arg=False)
    
    assert len(mw.param_df) == 2
    assert mw.param_df.loc["a","guess"] == 0
    assert mw.param_df.loc["b","guess"] == 2

    assert issubclass(type(mw),ModelWrapper)

    # send in dict of fit parameters
    mw = wrap_function(some_function=model_to_test_wrap,
                       fit_parameters={"a":{"guess":20}},
                       non_fit_kwargs=None,
                       vector_first_arg=False)
    
    assert len(mw.param_df) == 1
    assert mw.param_df.loc["a","guess"] == 20

    assert issubclass(type(mw),ModelWrapper)

    # send in dataframe of fit parameters
    df = pd.DataFrame({"name":["a"],
                       "guess":[20]})
    
    mw = wrap_function(some_function=model_to_test_wrap,
                       fit_parameters=df,
                       non_fit_kwargs=None,
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
                        non_fit_kwargs=None,
                        vector_first_arg=False)


    # send in spreadsheet file of fit parameters
    df = pd.DataFrame({"name":["a"],
                       "guess":[20]})
    df.to_csv("dataframe.csv")
    
    mw = wrap_function(some_function=model_to_test_wrap,
                       fit_parameters="dataframe.csv",
                       non_fit_kwargs=None,
                       vector_first_arg=False)
    
    assert len(mw.param_df) == 1
    assert mw.param_df.loc["a","guess"] == 20

    assert issubclass(type(mw),ModelWrapper)

    with pytest.raises(ValueError):
        mw = wrap_function(some_function=model_to_test_wrap,
                           fit_parameters=1,
                           non_fit_kwargs=None,
                           vector_first_arg=False)

    # send in non-default non_fit_kwargs
    mw = wrap_function(some_function=model_to_test_wrap,
                       fit_parameters=["a","b"],
                       non_fit_kwargs={"c":20},
                       vector_first_arg=False)
    
    assert len(mw.param_df) == 2
    assert mw.param_df.loc["a","guess"] == 0
    assert mw.param_df.loc["b","guess"] == 2
    assert len(mw._other_arguments) == 3
    assert mw._other_arguments["c"] == 20
    assert mw._other_arguments["d"] == "test"
    assert mw._other_arguments["e"] == 3

    # send in lots of non-default non_fit_kwargs
    mw = wrap_function(some_function=model_to_test_wrap,
                       fit_parameters=None,
                       non_fit_kwargs={"b":30,"c":20,"d":{"x":10},"e":str},
                       vector_first_arg=False)
    
    assert len(mw.param_df) == 1
    assert mw.param_df.loc["a","guess"] == 0
    assert len(mw._other_arguments) == 4
    assert mw._other_arguments["b"] == 30
    assert mw._other_arguments["c"] == 20
    assert mw._other_arguments["d"]["x"] == 10
    assert mw._other_arguments["e"] is str

    # kwargs! 
    def fcn_with_kwargs(a,b,**kwargs): pass
    
    mw = wrap_function(some_function=fcn_with_kwargs,
                       fit_parameters=["x","y","z"],
                       non_fit_kwargs={"b":np.nan,"c":20,"d":{"x":10},"e":str},
                       vector_first_arg=False)
    assert len(mw.param_df) == 3
    assert mw.param_df.loc["x","guess"] == 0
    assert mw.param_df.loc["y","guess"] == 0
    assert mw.param_df.loc["z","guess"] == 0
    
    assert len(mw._other_arguments) == 5
    assert mw._other_arguments["a"] is None
    assert np.isnan(mw._other_arguments["b"])
    assert mw._other_arguments["c"] == 20
    assert mw._other_arguments["d"]["x"] == 10
    assert mw._other_arguments["e"] is str

    # send in some bad non_fit_kwargs
    with pytest.raises(ValueError):
        mw = wrap_function(some_function=fcn_with_kwargs,
                        fit_parameters=["x","y","z"],
                        non_fit_kwargs="not_a_dict",
                        vector_first_arg=False)
        
    # send in some bad non_fit_kwargs
    with pytest.raises(ValueError):
        mw = wrap_function(some_function=fcn_with_kwargs,
                        fit_parameters=["x","y","z"],
                        non_fit_kwargs=["a","b","c"],
                        vector_first_arg=False)

    
    # --------------------------------------------------------------
    # vector_first_arg is True

    def model_to_test_wrap(some_vector,q=3): return np.sum(some_vector)

    # basic test. will die because VectorModelWrapper requires non-None 
    # fit_parameters
    with pytest.raises(ValueError):
        mw = wrap_function(some_function=model_to_test_wrap,
                        fit_parameters=None,
                        non_fit_kwargs=None,
                        vector_first_arg=True)
    
    # send in list of fit parameters
    mw = wrap_function(some_function=model_to_test_wrap,
                       fit_parameters=["a","b"],
                       non_fit_kwargs=None,
                       vector_first_arg=True)
    
    assert len(mw.param_df) == 2
    assert mw.param_df.loc["a","guess"] == 0
    assert mw.param_df.loc["b","guess"] == 0

    assert issubclass(type(mw),VectorModelWrapper)

    # send in dict of fit parameters
    mw = wrap_function(some_function=model_to_test_wrap,
                       fit_parameters={"a":{"guess":20}},
                       non_fit_kwargs=None,
                       vector_first_arg=True)
    
    assert len(mw.param_df) == 1
    assert mw.param_df.loc["a","guess"] == 20

    assert issubclass(type(mw),VectorModelWrapper)

    # send in dataframe of fit parameters
    df = pd.DataFrame({"name":["a"],
                       "guess":[20]})
    
    mw = wrap_function(some_function=model_to_test_wrap,
                       fit_parameters=df,
                       non_fit_kwargs=None,
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
                        non_fit_kwargs=None,
                        vector_first_arg=True)


    # send in spreadsheet file of fit parameters
    df = pd.DataFrame({"name":["a"],
                       "guess":[20]})
    df.to_csv("dataframe.csv")
    
    mw = wrap_function(some_function=model_to_test_wrap,
                       fit_parameters="dataframe.csv",
                       non_fit_kwargs=None,
                       vector_first_arg=True)
    
    assert len(mw.param_df) == 1
    assert mw.param_df.loc["a","guess"] == 20

    assert issubclass(type(mw),VectorModelWrapper)

    with pytest.raises(ValueError):
        mw = wrap_function(some_function=model_to_test_wrap,
                           fit_parameters=1,
                           non_fit_kwargs=None,
                           vector_first_arg=True)


    
    # send in non-default non_fit_kwargs
    mw = wrap_function(some_function=model_to_test_wrap,
                       fit_parameters=["a","b"],
                       non_fit_kwargs={"q":20},
                       vector_first_arg=True)
    
    assert len(mw.param_df) == 2
    assert mw.param_df.loc["a","guess"] == 0
    assert mw.param_df.loc["b","guess"] == 0
    assert len(mw._other_arguments) == 1
    assert mw._other_arguments["q"] == 20
    


    # kwargs! 
    def fcn_with_kwargs(a,b,**kwargs): pass
    
    mw = wrap_function(some_function=fcn_with_kwargs,
                       fit_parameters=["x","y","z"],
                       non_fit_kwargs={"b":np.nan,"c":20,"d":{"x":10},"e":str},
                       vector_first_arg=True)
    
    assert len(mw.param_df) == 3
    assert mw.param_df.loc["x","guess"] == 0
    assert mw.param_df.loc["y","guess"] == 0
    assert mw.param_df.loc["z","guess"] == 0
    
    assert len(mw._other_arguments) == 4
    assert np.isnan(mw._other_arguments["b"])
    assert mw._other_arguments["c"] == 20
    assert mw._other_arguments["d"]["x"] == 10
    assert mw._other_arguments["e"] is str

    # send in some bad non_fit_kwargs
    with pytest.raises(ValueError):
        mw = wrap_function(some_function=fcn_with_kwargs,
                        fit_parameters=["x","y","z"],
                        non_fit_kwargs="not_a_dict",
                        vector_first_arg=True)
        
    # send in some bad non_fit_kwargs
    with pytest.raises(ValueError):
        mw = wrap_function(some_function=fcn_with_kwargs,
                        fit_parameters=["x","y","z"],
                        non_fit_kwargs=["a","b","c"],
                        vector_first_arg=True)

    
   # --------------------------------------------------------------
    # vector_first_arg is stupid

    def model_to_test_wrap(some_vector,q=3): return np.sum(some_vector)

    # basic test. will die because VectorModelWrapper requires non-None 
    # fit_parameters
    with pytest.raises(ValueError):
        mw = wrap_function(some_function=model_to_test_wrap,
                           fit_parameters=["a"],
                           non_fit_kwargs=None,
                           vector_first_arg="stupid")

    os.chdir(cwd)