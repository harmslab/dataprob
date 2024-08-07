

import pandas as pd
import numpy as np

from dataprob.check import check_bool
from dataprob.check import check_float

def _read_spreadsheet(spreadsheet):

    # If this is a string, try to load it as a file
    if issubclass(type(spreadsheet),str):

        filename = spreadsheet

        ext = filename.split(".")[-1].strip().lower()

        if ext in ["xlsx","xls"]:
            df = pd.read_excel(filename)
        elif ext == "csv":
            df = pd.read_csv(filename,sep=",")
        elif ext == "tsv":
            df = pd.read_csv(filename,sep="\t")
        else:
            # Fall back -- try to guess delimiter
            df = pd.read_csv(filename,
                             sep=None,
                             engine="python",
                             encoding="utf-8-sig")

            

    # If this is a pandas dataframe, work in a copy of it.
    elif issubclass(type(spreadsheet),pd.DataFrame):
        df = spreadsheet.copy()

    # Otherwise, fail
    else:
        err = f"\n\n'spreadsheet' {spreadsheet} not recognized. Should be the\n"
        err += "filename of a spreadsheet or a pandas dataframe.\n"
        raise ValueError(err)
    
    return df

def load_param_spreadsheet(spreadsheet):

    df = _read_spreadsheet(spreadsheet=spreadsheet)

    if "param" not in df.columns:
        err = "param must be a column in the spreadsheet\n"
        raise ValueError(err)
    
    params = [str(p).strip() for p in df["param"]]
    if len(params) != len(np.unique(params)):
        err = "all entries in 'param' must be unique\n"
        raise ValueError(err)
    
    columns_in_df = set(df.columns)

    float_columns = ["guess",
                     "prior_mean","prior_std",
                     "lower_bound","upper_bound"]
    bool_columns = ["fixed"]
    
    columns_to_look_for = float_columns[:]
    columns_to_look_for.extend(bool_columns)
    columns_to_look_for = set(columns_to_look_for)
    
    columns = list(columns_to_look_for.intersection(columns_in_df))
    if len(columns) == 0:
        err = "no recognized columns in the spreadsheet.\n"
        raise ValueError(err)

    out = {}
    for i in range(len(df.index)):
        p = params[i]
        out[p] = {}
        for c in columns:
            if c in float_columns:
                v = check_float(value=df.loc[df.index[i],c],
                                variable_name=f"column '{c}'",
                                allow_nan=True)
            else:
                v = check_bool(value=df.loc[df.index[i],c],
                                variable_name=f"column '{c}'")

            out[p][c] = v

    for p in out:

        if "upper_bound" in out[p]:
            if np.isnan(out[p]["upper_bound"]):
                out[p]["upper_bound"] = np.inf
        if "lower_bound" in out[p]:
            if np.isnan(out[p]["lower_bound"]):
                out[p]["lower_bound"] = -np.inf

    return out
        


    
