
import pytest

from dataprob.model_wrapper.read_spreadsheet import _read_spreadsheet
from dataprob.model_wrapper.read_spreadsheet import _cleanup_guess
from dataprob.model_wrapper.read_spreadsheet import _cleanup_fixed
from dataprob.model_wrapper.read_spreadsheet import _cleanup_bounds
from dataprob.model_wrapper.read_spreadsheet import _cleanup_priors
from dataprob.model_wrapper.read_spreadsheet import load_param_spreadsheet

import numpy as np
import pandas as pd


def test__read_spreadsheet(spreadsheets):

    expected_columns = ['param',
                        'guess',
                        'lower_bound',
                        'upper_bound',
                        'prior_mean',
                        'prior_std',
                        'fixed']

    expected_values = {"param":["K1","K2","K3"],
                       "guess":[1e7,1e-6,1],
                       #"lower_bound":[pd.NA,pd.NA,-1], ## <- check below
                       #"upper_bound":[pd.NA,pd.NA,2],
                       #"prior_mean":[1e6,pd.NA,1e-6],
                       #"prior_std":[10,pd.NA,1000],
                       "fixed":[True,False,False]}


    # Check excel read
    xlsx = spreadsheets["basic-spreadsheet.xlsx"]
    df = _read_spreadsheet(spreadsheet=xlsx)
    
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
    df = _read_spreadsheet(spreadsheet=csv)
    
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
    df = _read_spreadsheet(spreadsheet=tsv)
    
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
    df = _read_spreadsheet(spreadsheet=txt)
    
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

    df = _read_spreadsheet(spreadsheet=df_in)
    
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
        _read_spreadsheet(spreadsheet=1)

    # file not found error
    with pytest.raises(FileNotFoundError):
        _read_spreadsheet(spreadsheet="not_a_file.txt")

def test__cleanup_guess():

    out = {"K1":{"guess":1.0}}
    out = _cleanup_guess(out)
    assert out["K1"]["guess"] == 1.0

    out = {"K1":{"not_guess":1.0}}
    out = _cleanup_guess(out)
    assert out["K1"]["not_guess"] == 1.0

    out = {"K1":{"guess":np.nan}}
    with pytest.raises(ValueError):
        out = _cleanup_guess(out)
    
    out = {"K1":{"guess":pd.NA}}
    with pytest.raises(ValueError):
        out = _cleanup_guess(out)

def test__cleanup_fixed():
    
    out = {"K1":{"fixed":True}}
    out = _cleanup_fixed(out)
    assert out["K1"]["fixed"] is True
    assert issubclass(type(out["K1"]["fixed"]),bool)

    out = {"K1":{"fixed":False}}
    out = _cleanup_fixed(out)
    assert out["K1"]["fixed"] is False
    assert issubclass(type(out["K1"]["fixed"]),bool)

    out = {"K1":{"not_fixed":1.0}}
    out = _cleanup_fixed(out)
    assert out["K1"]["not_fixed"] == 1.0

    out = {"K1":{"fixed":1.5}}
    with pytest.raises(ValueError):
        out = _cleanup_fixed(out)

    out = {"K1":{"fixed":np.nan}}
    with pytest.raises(ValueError):
        out = _cleanup_fixed(out)
    
    out = {"K1":{"fixed":pd.NA}}
    with pytest.raises(ValueError):
        out = _cleanup_fixed(out)

def test__cleanup_bounds():

    out = {"K1":{"upper_bound":1}}
    out = _cleanup_bounds(out)
    assert "upper_bound" not in out["K1"]
    assert out["K1"]["bounds"][1] == 1
    assert out["K1"]["bounds"][0] < 0
    assert np.isinf(out["K1"]["bounds"][0])

    out = {"K1":{"lower_bound":1}}
    out = _cleanup_bounds(out)
    assert "lower_bound" not in out["K1"]
    assert out["K1"]["bounds"][0] == 1
    assert out["K1"]["bounds"][1] > 0
    assert np.isinf(out["K1"]["bounds"][1])

    out = {"K1":{"lower_bound":1,"upper_bound":2}}
    out = _cleanup_bounds(out)
    assert "upper_bound" not in out["K1"]
    assert "lower_bound" not in out["K1"]
    assert out["K1"]["bounds"][0] == 1
    assert out["K1"]["bounds"][1] == 2 

    out = {"K1":{"lower_bound":np.nan,"upper_bound":2}}
    out = _cleanup_bounds(out)
    assert "upper_bound" not in out["K1"]
    assert "lower_bound" not in out["K1"]
    assert out["K1"]["bounds"][0] < 0
    assert np.isinf(out["K1"]["bounds"][0])
    assert out["K1"]["bounds"][1] == 2 

    out = {"K1":{"not_bounds":1}}
    out = _cleanup_bounds(out)
    assert "bounds" not in out["K1"]
    assert "not_bounds" in out["K1"]
    assert out["K1"]["not_bounds"] == 1
    
def test__cleanup_priors():

    out = {"K1":{"prior_mean":10,"prior_std":1}}
    out = _cleanup_priors(out)
    assert "prior_mean" not in out["K1"]
    assert "prior_std" not in out["K1"]
    assert "prior" in out["K1"]
    assert np.array_equal(out["K1"]["prior"],[10,1])

    out = {"K1":{"prior_mean":10}}
    with pytest.raises(ValueError):
        out = _cleanup_priors(out)
    
    out = {"K1":{"prior_std":1}}
    with pytest.raises(ValueError):
        out = _cleanup_priors(out)

    out = {"K1":{"prior_mean":10,"prior_std":np.nan}}
    with pytest.raises(ValueError):
        out = _cleanup_priors(out)

    out = {"K1":{"prior_mean":pd.NA,"prior_std":1}}
    with pytest.raises(ValueError):
        out = _cleanup_priors(out)

    out = {"K1":{"not_priors":1}}
    out = _cleanup_priors(out)
    assert "bounds" not in out["K1"]
    assert "not_priors" in out["K1"]
    assert out["K1"]["not_priors"] == 1

    
def test_load_param_spreadsheet(spreadsheets):

    xlsx = spreadsheets["basic-spreadsheet.xlsx"]
    df = _read_spreadsheet(spreadsheet=xlsx)
    
    # Spreadsheets cover all scenarios of bool and float reads
    out = load_param_spreadsheet(df)

    assert out["K1"]["guess"] == 1e7
    assert np.isinf(out["K1"]["bounds"][0])
    assert out["K1"]["bounds"][0] < 0
    assert np.isinf(out["K1"]["bounds"][1])
    assert out["K1"]["prior"][0] == 1e6
    assert out["K1"]["prior"][1] == 10
    assert out["K1"]["fixed"] is True

    assert out["K2"]["guess"] == 1e-6
    assert np.isinf(out["K2"]["bounds"][0])
    assert out["K2"]["bounds"][0] < 0
    assert np.isinf(out["K2"]["bounds"][1])
    assert np.isnan(out["K2"]["prior"][0])
    assert np.isnan(out["K2"]["prior"][1])
    assert out["K2"]["fixed"] is False

    assert out["K3"]["guess"] == 1
    assert out["K3"]["bounds"][0] == -1
    assert out["K3"]["bounds"][1] == 2
    assert out["K3"]["prior"][0] == 1e-6
    assert out["K3"]["prior"][1] == 1000
    assert out["K3"]["fixed"] is False

    # Test direct read from file
    out = load_param_spreadsheet(spreadsheet=spreadsheets["basic-spreadsheet.xlsx"])
    
    assert out["K1"]["guess"] == 1e7
    assert np.isinf(out["K1"]["bounds"][0])
    assert out["K1"]["bounds"][0] < 0
    assert np.isinf(out["K1"]["bounds"][1])
    assert out["K1"]["prior"][0] == 1e6
    assert out["K1"]["prior"][1] == 10
    assert out["K1"]["fixed"] is True

    assert out["K2"]["guess"] == 1e-6
    assert np.isinf(out["K2"]["bounds"][0])
    assert out["K2"]["bounds"][0] < 0
    assert np.isinf(out["K2"]["bounds"][1])
    assert np.isnan(out["K2"]["prior"][0])
    assert np.isnan(out["K2"]["prior"][1])
    assert out["K2"]["fixed"] is False

    assert out["K3"]["guess"] == 1
    assert out["K3"]["bounds"][0] == -1
    assert out["K3"]["bounds"][1] == 2
    assert out["K3"]["prior"][0] == 1e-6
    assert out["K3"]["prior"][1] == 1000
    assert out["K3"]["fixed"] is False

    # make dataframe where we drop "param" column
    new_df = df.copy().loc[:,df.columns[1:]]
    assert not "param" in new_df.columns
    with pytest.raises(ValueError):
        load_param_spreadsheet(spreadsheet=new_df)

    # make dataframe where two params have the same name
    new_df = df.copy()
    new_df["param"] = ["K1","K1","K3"]
    with pytest.raises(ValueError):
        load_param_spreadsheet(spreadsheet=new_df)

    # send in column with a bad fixed value

    # make sure we can read spreadsheet
    df = _read_spreadsheet(spreadsheet=spreadsheets["bad-fixed.xlsx"])
    with pytest.raises(ValueError):
        load_param_spreadsheet(spreadsheet=spreadsheets["bad-fixed.xlsx"])

    # send in column with a bad guess value
    
    # make sure we can read spreadsheet
    df = _read_spreadsheet(spreadsheet=spreadsheets["bad-guess.xlsx"])
    with pytest.raises(ValueError):
        load_param_spreadsheet(spreadsheet=spreadsheets["bad-guess.xlsx"])

    # Send in a dataframe with only param but no other columns
    df = pd.DataFrame({"param":["K1","K2"],"other_column":[1,2]})
    with pytest.raises(ValueError):
        load_param_spreadsheet(spreadsheet=df)
    
    # make sure this is fixed by adding column of interest (also making sure 
    # the code can handle extra columns)
    df["guess"] = [10,11]
    out = load_param_spreadsheet(spreadsheet=df)
    assert out["K1"]["guess"] == 10
    assert out["K2"]["guess"] == 11

    # only prior_mean (should die)
    df = pd.DataFrame({"param":["K1","K2"],
                       "prior_mean":[1,1]})
    with pytest.raises(ValueError):
        load_param_spreadsheet(spreadsheet=df)

    # only prior_std (should die)
    df = pd.DataFrame({"param":["K1","K2"],
                       "prior_std":[1,1]})
    with pytest.raises(ValueError):
        load_param_spreadsheet(spreadsheet=df)

    # only upper_bound (should be fine, set lower_bound to -inf)
    df = pd.DataFrame({"param":["K1","K2"],
                       "upper_bound":[1,1]})
    
    out = load_param_spreadsheet(spreadsheet=df)
    assert out["K1"]["bounds"][1] == 1
    assert out["K1"]["bounds"][0] < 0
    assert np.isinf(out["K1"]["bounds"][0])
    assert out["K2"]["bounds"][1] == 1
    assert out["K2"]["bounds"][0] < 0
    assert np.isinf(out["K2"]["bounds"][0])

    # only lower_bound (should be fine, set upper_bound to inf)
    df = pd.DataFrame({"param":["K1","K2"],
                       "lower_bound":[1,1]})
    out = load_param_spreadsheet(spreadsheet=df)
    assert out["K1"]["bounds"][0] == 1
    assert out["K1"]["bounds"][1] > 0
    assert np.isinf(out["K1"]["bounds"][1])
    assert out["K2"]["bounds"][0] == 1
    assert out["K2"]["bounds"][1] > 0
    assert np.isinf(out["K2"]["bounds"][1])

    # upper and lower both nan (should be fine, set lower_bound to -inf and upper_bound to inf)
    df = pd.DataFrame({"param":["K1","K2"],
                       "lower_bound":[pd.NA,pd.NA],
                       "upper_bound":[np.nan,np.nan]})
    out = load_param_spreadsheet(spreadsheet=df)
    assert out["K1"]["bounds"][0] < 0
    assert out["K1"]["bounds"][1] > 0
    assert np.isinf(out["K1"]["bounds"][0])
    assert np.isinf(out["K1"]["bounds"][1])
    assert out["K2"]["bounds"][0] < 0
    assert out["K2"]["bounds"][1] > 0
    assert np.isinf(out["K2"]["bounds"][0])
    assert np.isinf(out["K2"]["bounds"][1])
    
    # nan guess (should die)
    df = pd.DataFrame({"param":["K1","K2"],
                       "guess":[np.nan,np.nan]})
    with pytest.raises(ValueError):
        load_param_spreadsheet(spreadsheet=df)






    
    

    