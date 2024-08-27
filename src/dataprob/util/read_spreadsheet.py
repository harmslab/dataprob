"""
Code for dealing with spreadsheets in a user-friendly way. 
"""

import pandas as pd

def read_spreadsheet(spreadsheet):
    """
    Read a spreadsheet. Use pandas to read files of various types or, if 
    spreadsheet is already a dataframe, return a copy of the dataframe. 
    
    Parameters
    ----------
    spreadsheet : str or pandas.DataFrame
        filename of spreadsheet to read or dataframe
    
    Returns
    -------
    df : pandas.DataFrame
        pandas dataframe read from filename or copied from input dataframe
    """

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

    # If this is a pandas dataframe, create a copy of it.
    elif issubclass(type(spreadsheet),pd.DataFrame):
        df = spreadsheet.copy()

    # Otherwise, fail
    else:
        err = f"\n\n'spreadsheet' {spreadsheet} not recognized. Should be the\n"
        err += "filename of a spreadsheet or a pandas dataframe.\n"
        raise ValueError(err)
    
    return df
