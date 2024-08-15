
import pytest

from dataprob.model_wrapper._dataframe_processing import _check_name
from dataprob.model_wrapper._dataframe_processing import _build_columns
from dataprob.model_wrapper._dataframe_processing import _check_bounds
from dataprob.model_wrapper._dataframe_processing import _check_guesses
from dataprob.model_wrapper._dataframe_processing import _check_priors

from dataprob.model_wrapper._dataframe_processing import _df_to_dict

from dataprob.model_wrapper._dataframe_processing import read_spreadsheet
from dataprob.model_wrapper._dataframe_processing import validate_dataframe
from dataprob.model_wrapper._dataframe_processing import param_into_existing

import numpy as np
import pandas as pd

def test__check_name():

    # basic run, make sure working on a copy
    good_df = pd.DataFrame({"name":["K1","K2","K3"],
                            "test":[1,2,3]})
    out_df = _check_name(param_df=good_df,
                         param_in_order=["K1","K2","K3"])
    assert good_df is not out_df

    # no name column
    bad_df = good_df.drop(columns=["name"])
    with pytest.raises(ValueError):
        _check_name(param_df=bad_df,
                    param_in_order=["K1","K2","K3"])
        
    # name not in df
    with pytest.raises(ValueError):
        _check_name(param_df=good_df,
                    param_in_order=["K1","K2","K3","K4"])
        
    # name not in param_in_order
    with pytest.raises(ValueError):
        _check_name(param_df=good_df,
                    param_in_order=["K1","K2"])
        
    # not unique parameter names
    bad_df = pd.DataFrame({"name":["K1","K2","K2"],
                            "test":[1,2,3]})
    with pytest.raises(ValueError):
        _check_name(param_df=bad_df,
                    param_in_order=["K1","K2","K2"])
    
    # check index assignment and ordering
    good_df = pd.DataFrame({"name":["K3","K2","K1"],
                            "test":[1,2,3]})
    out_df = _check_name(param_df=good_df,
                         param_in_order=["K1","K2","K3"])
    assert out_df.loc["K1","test"] == 3
    assert out_df.loc["K2","test"] == 2
    assert out_df.loc["K3","test"] == 1
    assert np.array_equal(out_df.index,["K1","K2","K3"])

def test__build_columns():

    # Test default dataframe column building
    df = pd.DataFrame({"name":["a","b"]})
    out_df = _build_columns(param_df=df,
                            default_guess=10)
    
    assert np.array_equal(out_df.columns,["name","guess","fixed","lower_bound",
                                          "upper_bound","prior_mean","prior_std"])
    assert np.array_equal(out_df["name"],["a","b"])
    assert np.array_equal(out_df["guess"],[10,10])
    assert np.array_equal(out_df["fixed"],[False,False])
    assert np.array_equal(out_df["lower_bound"],[-np.inf,-np.inf])
    assert np.array_equal(out_df["upper_bound"],[np.inf,np.inf])
    assert np.sum(np.isnan(out_df["prior_mean"])) == 2
    assert np.sum(np.isnan(out_df["prior_std"])) == 2

    # make sure existing columns are left intact
    df = pd.DataFrame({"name":["a","b"],
                       "guess":[20,20],
                       "fixed":[False,True],
                       "lower_bound":[-200,-200],
                       "upper_bound":[200,200],
                       "prior_mean":[np.nan,20],
                       "prior_std":[np.nan,10]})
    out_df = _build_columns(param_df=df,
                            default_guess=10)
    assert np.array_equal(out_df.columns,["name","guess","fixed","lower_bound",
                                          "upper_bound","prior_mean","prior_std"])
    assert np.array_equal(out_df["name"],["a","b"])
    assert np.array_equal(out_df["guess"],[20,20])
    assert np.array_equal(out_df["fixed"],[False,True])
    assert np.array_equal(out_df["lower_bound"],[-200,-200])
    assert np.array_equal(out_df["upper_bound"],[200,200])
    assert np.array_equal(out_df["prior_mean"],[np.nan,20],equal_nan=True)
    assert np.array_equal(out_df["prior_std"],[np.nan,10],equal_nan=True)

    # float coercion check
    df = pd.DataFrame({"name":["a","b"],
                       "guess":["20",20.0],
                       "fixed":[False,True],
                       "lower_bound":["-200",-200],
                       "upper_bound":["200",200],
                       "prior_mean":[np.nan,"20"],
                       "prior_std":[np.nan,"10"]})
    out_df = _build_columns(param_df=df,
                            default_guess=10)
    assert np.array_equal(out_df.columns,["name","guess","fixed","lower_bound",
                                          "upper_bound","prior_mean","prior_std"])
    assert np.array_equal(out_df["name"],["a","b"])
    assert np.array_equal(out_df["guess"],[20,20])
    assert np.array_equal(out_df["fixed"],[False,True])
    assert np.array_equal(out_df["lower_bound"],[-200.0,-200.0])
    assert np.array_equal(out_df["upper_bound"],[200,200])
    assert np.array_equal(out_df["prior_mean"],[np.nan,20],equal_nan=True)
    assert np.array_equal(out_df["prior_std"],[np.nan,10],equal_nan=True)
    
    # We sent in guess above as an integer. Make sure it's being properly
    # coerced to a float. We sent in lower_bound as a float. It should also
    # be a float. 
    assert np.issubdtype(out_df["guess"].dtype, np.floating)
    assert np.issubdtype(out_df["lower_bound"].dtype, np.floating)
    
    # bad float coercion check
    for bad_key in ["guess","lower_bound","upper_bound","prior_mean","prior_std"]:
        df = pd.DataFrame({"name":["a","b"],
                           bad_key:["fail",1]})
        with pytest.raises(ValueError):
            out_df = _build_columns(param_df=df,
                            default_guess=10)
            
    # verify nan coercion because other functions assume nan rather than None or 
    # pd.NA. This replicates what happens in the code. Not optimal, but should 
    # at least catch changes due to altered pandas or numpy implementations
    df = pd.DataFrame({"x":[np.nan,pd.NA,None,1]})
    # raise an error because not a strictly float column
    with pytest.raises(TypeError):
        np.isnan(df["x"]) 
    df["x"] = pd.to_numeric(df["x"])
    # works now because pd.NA and None were coerced to np.nan
    assert np.array_equal(np.isnan(df["x"]),[True,True,True,False])
    
    # fixed coercion
    df = pd.DataFrame({"name":["a","b"],
                       "fixed":[0,"True"]})
    out_df = _build_columns(param_df=df,
                            default_guess=10)
    assert np.array_equal(out_df["fixed"],[False,True])

    # bad fixed coercion
    df = pd.DataFrame({"name":["a","b"],
                       "fixed":[pd.NA,"True"]})
    with pytest.raises(ValueError):
        out_df = _build_columns(param_df=df,
                                default_guess=10)
        
def test__check_bounds():

    # check automatic nan assignment
    df = pd.DataFrame({"name":["a","b","c","d"],
                       "lower_bound":[np.nan,pd.NA,None,-1],
                       "upper_bound":[np.nan,pd.NA,None,1]})
    out_df = _check_bounds(param_df=df)
    assert np.array_equal(out_df["lower_bound"],[-np.inf,-np.inf,-np.inf,-1])
    assert np.array_equal(out_df["upper_bound"],[np.inf,np.inf,np.inf,1])

    # Same values
    for same in [-np.inf,-10.0,0.0,10.0,np.inf]:
        print("checking same: ")
        df = pd.DataFrame({"name":["a"],
                        "lower_bound":[same],
                        "upper_bound":[same]})
        with pytest.raises(ValueError):
            _check_bounds(df)

    # flipped values (floats)
    df = pd.DataFrame({"name":["a"],
                        "lower_bound":[10.0],
                        "upper_bound":[0.0]})
    with pytest.raises(ValueError):
        _check_bounds(df)

    # flipped values (float vs inf)
    df = pd.DataFrame({"name":["a"],
                        "lower_bound":[np.inf],
                        "upper_bound":[0.0]})
    with pytest.raises(ValueError):
        _check_bounds(df)

        # flipped values (float vs inf)
    df = pd.DataFrame({"name":["a"],
                        "lower_bound":[10.0],
                        "upper_bound":[-np.inf]})
    with pytest.raises(ValueError):
        _check_bounds(df)

def test__check_guesses():

    test_df = _build_columns(pd.DataFrame({"name":["a","b"]}),
                             default_guess=10)
    
    out_df = _check_guesses(test_df)
    assert np.array_equal(out_df["guess"],[10,10])

    # above lower bound
    df = test_df.copy()
    df["lower_bound"] = 20
    with pytest.raises(ValueError):
        _check_guesses(df)

    # below upper bound
    df = test_df.copy()
    df["upper_bound"] = 1
    with pytest.raises(ValueError):
        _check_guesses(df)

    # nan guess
    df = test_df.copy()
    df["guess"] = [np.nan,np.nan]
    with pytest.raises(ValueError):
        _check_guesses(df)

def test__check_priors():
    
    test_df = pd.DataFrame({"name":["a","b"],
                            "prior_mean":[np.nan,np.nan],
                            "prior_std":[np.nan,np.nan]})
    
    # both nan (okay)
    df = test_df.copy()
    df["prior_mean"] = [np.nan,np.nan]
    df["prior_std"] = [np.nan,np.nan]
    out_df = _check_priors(test_df)
    assert np.sum(np.isnan(out_df["prior_mean"])) == 2
    assert np.sum(np.isnan(out_df["prior_std"])) == 2

    # mean nan only (not okay)
    df = test_df.copy()
    df["prior_mean"] = [1.0,np.nan]
    df["prior_std"] = [np.nan,np.nan]
    with pytest.raises(ValueError):
        _check_priors(df)

    # std nan only (not okay)
    df = test_df.copy()
    df["prior_mean"] = [np.nan,np.nan]
    df["prior_std"] = [1.0,np.nan]
    with pytest.raises(ValueError):
        _check_priors(df)

    # both non-nan (okay)
    df = test_df.copy()
    df["prior_mean"] = [1.0,np.nan]
    df["prior_std"] = [1.0,np.nan]
    out_df = _check_priors(df)
    assert np.array_equal(out_df["prior_mean"],[1.0,np.nan],equal_nan=True)
    assert np.array_equal(out_df["prior_std"],[1.0,np.nan],equal_nan=True)

    # negative std, fail
    df = test_df.copy()
    df["prior_mean"] = [1.0,np.nan]
    df["prior_std"] = [-1.0,np.nan]
    with pytest.raises(ValueError):
        _check_priors(df)

    # inf bad, fail
    df = test_df.copy()
    df["prior_mean"] = [np.inf,np.nan]
    df["prior_std"] = [1,np.nan]
    with pytest.raises(ValueError):
        _check_priors(df)

    # inf bad, fail
    df = test_df.copy()
    df["prior_mean"] = [1,np.nan]
    df["prior_std"] = [np.inf,np.nan]
    with pytest.raises(ValueError):
        _check_priors(df)

    # inf bad, fail
    df = test_df.copy()
    df["prior_mean"] = [np.inf,np.nan]
    df["prior_std"] = [np.inf,np.nan]
    with pytest.raises(ValueError):
        _check_priors(df)
    
def test__df_to_dict():

    # name column check
    some_df = pd.DataFrame({"blah":[1,2]})
    with pytest.raises(ValueError):
        _df_to_dict(some_df)

    # validate string coercion
    some_df = pd.DataFrame({"name":[1,2]})
    out = _df_to_dict(some_df)
    assert len(out.keys()) == 2
    assert np.array_equal(list(out.keys()),["1","2"])

    # validate uniqueness check
    some_df = pd.DataFrame({"name":["test","test"]})
    with pytest.raises(ValueError):
        _df_to_dict(some_df)

    # should work
    some_df = pd.DataFrame({"name":["a","b"],
                            "test":[1,2],
                            "this":["c","d"]})
    some_dict = _df_to_dict(some_df)
    
    assert len(some_dict.keys()) == 2
    assert np.array_equal(list(some_dict.keys()),["a","b"])
    assert some_dict["a"]["test"] == 1
    assert some_dict["b"]["test"] == 2
    assert some_dict["a"]["this"] == "c"
    assert some_dict["b"]["this"] == "d"


def test_read_spreadsheet(spreadsheets):

    expected_columns = ['name',
                        'guess',
                        'lower_bound',
                        'upper_bound',
                        'prior_mean',
                        'prior_std',
                        'fixed']

    expected_values = {"name":["K1","K2","K3"],
                       "guess":[1e7,1e-6,1],
                       #"lower_bound":[pd.NA,pd.NA,-1], ## <- check below
                       #"upper_bound":[pd.NA,pd.NA,2],
                       #"prior_mean":[1e6,pd.NA,1e-6],
                       #"prior_std":[10,pd.NA,1000],
                       "fixed":[True,False,False]}


    # Check excel read
    xlsx = spreadsheets["basic-spreadsheet.xlsx"]
    df = read_spreadsheet(spreadsheet=xlsx)
    
    assert np.array_equal(df.columns,expected_columns)
    for e in expected_values:
        print("xlsx",e)
        assert np.array_equal(df[e],expected_values[e])

    assert np.array_equal(pd.isna(df["lower_bound"]),[True,True,False])
    assert df.loc[2,"lower_bound"] == -1

    assert np.array_equal(pd.isna(df["upper_bound"]),[True,True,False])
    assert df.loc[2,"upper_bound"] == 2

    assert np.array_equal(pd.isna(df["prior_mean"]),[False,True,False])
    assert np.array_equal(df.loc[[0,2],"prior_mean"],[1e6,1e-6])

    assert np.array_equal(pd.isna(df["prior_std"]),[False,True,False])
    assert np.array_equal(df.loc[[0,2],"prior_std"],[10,1000])

    # Check csv read
    csv = spreadsheets["basic-spreadsheet.csv"]
    df = read_spreadsheet(spreadsheet=csv)
    
    assert np.array_equal(df.columns,expected_columns)
    for e in expected_values:
        print("csv",e)
        assert np.array_equal(df[e],expected_values[e])

    assert np.array_equal(pd.isna(df["lower_bound"]),[True,True,False])
    assert df.loc[2,"lower_bound"] == -1

    assert np.array_equal(pd.isna(df["upper_bound"]),[True,True,False])
    assert df.loc[2,"upper_bound"] == 2

    assert np.array_equal(pd.isna(df["prior_mean"]),[False,True,False])
    assert np.array_equal(df.loc[[0,2],"prior_mean"],[1e6,1e-6])

    assert np.array_equal(pd.isna(df["prior_std"]),[False,True,False])
    assert np.array_equal(df.loc[[0,2],"prior_std"],[10,1000])


    # Check tsv read
    tsv = spreadsheets["basic-spreadsheet.tsv"]
    df = read_spreadsheet(spreadsheet=tsv)
    
    assert np.array_equal(df.columns,expected_columns)
    for e in expected_values:
        print("tsv",e)
        assert np.array_equal(df[e],expected_values[e])

    assert np.array_equal(pd.isna(df["lower_bound"]),[True,True,False])
    assert df.loc[2,"lower_bound"] == -1

    assert np.array_equal(pd.isna(df["upper_bound"]),[True,True,False])
    assert df.loc[2,"upper_bound"] == 2

    assert np.array_equal(pd.isna(df["prior_mean"]),[False,True,False])
    assert np.array_equal(df.loc[[0,2],"prior_mean"],[1e6,1e-6])

    assert np.array_equal(pd.isna(df["prior_std"]),[False,True,False])
    assert np.array_equal(df.loc[[0,2],"prior_std"],[10,1000])

    # Check txt read
    txt = spreadsheets["basic-spreadsheet.txt"]
    df = read_spreadsheet(spreadsheet=txt)
    
    assert np.array_equal(df.columns,expected_columns)
    for e in expected_values:
        print("txt",e)
        assert np.array_equal(df[e],expected_values[e])

    assert np.array_equal(pd.isna(df["lower_bound"]),[True,True,False])
    assert df.loc[2,"lower_bound"] == -1

    assert np.array_equal(pd.isna(df["upper_bound"]),[True,True,False])
    assert df.loc[2,"upper_bound"] == 2

    assert np.array_equal(pd.isna(df["prior_mean"]),[False,True,False])
    assert np.array_equal(df.loc[[0,2],"prior_mean"],[1e6,1e-6])

    assert np.array_equal(pd.isna(df["prior_std"]),[False,True,False])
    assert np.array_equal(df.loc[[0,2],"prior_std"],[10,1000])

    # Check pass through
    xlsx = spreadsheets["basic-spreadsheet.xlsx"]
    df_in = pd.read_excel(xlsx)

    df = read_spreadsheet(spreadsheet=df_in)
    
    assert np.array_equal(df.columns,expected_columns)
    for e in expected_values:
        print("xlsx",e)
        assert np.array_equal(df[e],expected_values[e])

    assert np.array_equal(pd.isna(df["lower_bound"]),[True,True,False])
    assert df.loc[2,"lower_bound"] == -1

    assert np.array_equal(pd.isna(df["upper_bound"]),[True,True,False])
    assert df.loc[2,"upper_bound"] == 2

    assert np.array_equal(pd.isna(df["prior_mean"]),[False,True,False])
    assert np.array_equal(df.loc[[0,2],"prior_mean"],[1e6,1e-6])

    assert np.array_equal(pd.isna(df["prior_std"]),[False,True,False])
    assert np.array_equal(df.loc[[0,2],"prior_std"],[10,1000])

    # send in something stupid
    with pytest.raises(ValueError):
        read_spreadsheet(spreadsheet=1)

    # file not found error
    with pytest.raises(FileNotFoundError):
        read_spreadsheet(spreadsheet="not_a_file.txt")

def test_validate_dataframe(spreadsheets):

    test_spreadsheet = spreadsheets["basic-spreadsheet.xlsx"]
    
    # validate dataframe check
    with pytest.raises(ValueError):
        validate_dataframe(param_df=test_spreadsheet,
                           param_in_order=["K1","K2","K3"],
                           default_guess=0)
        
    # All-in df check
    df = read_spreadsheet(test_spreadsheet)
    param_df = validate_dataframe(param_df=df,
                                  param_in_order=["K1","K2","K3"],
                                  default_guess=0)
    
    assert np.array_equal(param_df["name"],
                          ["K1","K2","K3"])
    assert np.array_equal(param_df["guess"],
                          [1e7,1e-6,1])
    assert np.array_equal(param_df["fixed"],
                          [True,False,False])

    assert np.array_equal(np.isinf(param_df["lower_bound"]),
                          [True,True,False])
    assert param_df.loc["K3","lower_bound"] == -1

    assert np.array_equal(np.isinf(param_df["upper_bound"]),
                          [True,True,False])
    assert param_df.loc["K3","upper_bound"] == 2

    assert np.array_equal(pd.isna(param_df["prior_mean"]),
                          [False,True,False])
    assert param_df.loc["K1","prior_mean"] == 1e6
    assert np.array_equal(pd.isna(param_df["prior_std"]),
                          [False,True,False])
    assert param_df.loc["K1","prior_std"] == 10

    
def test_param_into_existing():
    
    param_df = pd.DataFrame({"name":["a","b"],
                             "guess":[1,2]})
    param_df = validate_dataframe(param_df,
                                  param_in_order=["a","b"])
    assert np.array_equal(param_df["guess"],[1,2])

    # Bad input
    with pytest.raises(ValueError):
        param_into_existing(param_input="not_good",
                            param_df=param_df)
    
    # Bring in df
    param_input_df = pd.DataFrame({"name":["a"],
                                   "guess":[10]})
    start_df = param_df.copy()
    end_df = param_into_existing(param_input=param_input_df,
                                 param_df=param_df)
    assert end_df is not start_df # working on a copy
    assert np.array_equal(end_df["name"],["a","b"])
    assert np.array_equal(end_df["guess"],[10,2])

    # bring in dict
    param_input_dict = {"a":{"guess":10}}
    start_df = param_df.copy()
    end_df = param_into_existing(param_input=param_input_dict,
                                 param_df=param_df)
    assert end_df is not start_df # working on a copy
    assert np.array_equal(end_df["name"],["a","b"])
    assert np.array_equal(end_df["guess"],[10,2])

    # name not in param_df
    param_input_dict_bad = {"c":{"guess":10}}
    with pytest.raises(ValueError):
        param_into_existing(param_input=param_input_dict_bad,
                            param_df=start_df)

    # Extra column
    param_input_df = pd.DataFrame({"name":["a"],
                                   "guess":[20],
                                   "extra_col":[5.0]})
    start_df = param_df.copy()
    end_df = param_into_existing(param_input=param_input_df,
                                 param_df=param_df)
    assert end_df is not start_df # working on a copy
    assert np.array_equal(end_df["name"],["a","b"])
    assert np.array_equal(end_df["guess"],[20,2])
    assert np.array_equal(end_df["extra_col"],[5,np.nan],equal_nan=True)
    
    # Extra column via dict
    param_input_dict = {"a":{"guess":30},
                        "b":{"extra_col":5.0}}
    start_df = param_df.copy()
    end_df = param_into_existing(param_input=param_input_dict,
                                 param_df=param_df)
    assert end_df is not start_df # working on a copy
    assert np.array_equal(end_df["name"],["a","b"])
    assert np.array_equal(end_df["guess"],[30,2])
    assert np.array_equal(end_df["extra_col"],[np.nan,5.0],equal_nan=True)
    