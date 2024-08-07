
import pytest

from dataprob.model_wrapper.read_spreadsheet import _read_spreadsheet
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

def test_load_param_spreadsheet(spreadsheets):

    xlsx = spreadsheets["basic-spreadsheet.xlsx"]
    df = _read_spreadsheet(spreadsheet=xlsx)
    
    # Spreadsheets cover all scenarios of bool and float reads
    out = load_param_spreadsheet(df)

    assert out["K1"]["guess"] == 1e7
    assert np.isinf(out["K1"]["lower_bound"])
    assert out["K1"]["lower_bound"] < 0
    assert np.isinf(out["K1"]["upper_bound"])
    assert out["K1"]["prior_mean"] == 1e6
    assert out["K1"]["prior_std"] == 10
    assert out["K1"]["fixed"] is True

    assert out["K2"]["guess"] == 1e-6
    assert np.isinf(out["K2"]["lower_bound"])
    assert out["K2"]["lower_bound"] < 0
    assert np.isinf(out["K2"]["upper_bound"])
    assert np.isnan(out["K2"]["prior_mean"])
    assert np.isnan(out["K2"]["prior_std"])
    assert out["K2"]["fixed"] is False

    assert out["K3"]["guess"] == 1
    assert out["K3"]["lower_bound"] == -1
    assert out["K3"]["upper_bound"] == 2
    assert out["K3"]["prior_mean"] == 1e-6
    assert out["K3"]["prior_std"] == 1000
    assert out["K3"]["fixed"] is False

    # Test direct read from file
    out = load_param_spreadsheet(spreadsheet=spreadsheets["basic-spreadsheet.xlsx"])
    
    assert out["K1"]["guess"] == 1e7
    assert np.isinf(out["K1"]["lower_bound"])
    assert out["K1"]["lower_bound"] < 0
    assert np.isinf(out["K1"]["upper_bound"])
    assert out["K1"]["prior_mean"] == 1e6
    assert out["K1"]["prior_std"] == 10
    assert out["K1"]["fixed"] is True

    assert out["K2"]["guess"] == 1e-6
    assert np.isinf(out["K2"]["lower_bound"])
    assert out["K2"]["lower_bound"] < 0
    assert np.isinf(out["K2"]["upper_bound"])
    assert np.isnan(out["K2"]["prior_mean"])
    assert np.isnan(out["K2"]["prior_std"])
    assert out["K2"]["fixed"] is False

    assert out["K3"]["guess"] == 1
    assert out["K3"]["lower_bound"] == -1
    assert out["K3"]["upper_bound"] == 2
    assert out["K3"]["prior_mean"] == 1e-6
    assert out["K3"]["prior_std"] == 1000
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

    
    

    



    
    

    