import pytest

from dataprob.util.read_spreadsheet import read_spreadsheet

import numpy as np
import pandas as pd

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
