

import pandas as pd
import numpy as np

from dataprob.check import check_bool
from dataprob.check import check_float

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

def _cleanup_guess(out):
    """
    If either a guess is specified, process it. 

    Parameters
    ----------
    out : dict
        dictionary of parameter values

    Returns
    -------
    out : dict
        dictionary of parameter values with bounds cleaned up
    """

    for p in out:
        if "guess" in out[p]:
            out[p]["guess"] = check_float(value=out[p]["guess"],
                                          variable_name=f"param {p} guess",
                                          allow_nan=False)
            
    return out

def _cleanup_fixed(out):
    """
    If fixed was specified, process it. 

    Parameters
    ----------
    out : dict
        dictionary of parameter values

    Returns
    -------
    out : dict
        dictionary of parameter values with fixed cleaned up
    """

    for p in out:
        if "fixed" in out[p]:
            out[p]["fixed"] = check_bool(value=out[p]["fixed"],
                                         variable_name=f"param {p} fixed")
            
    return out

def _cleanup_bounds(out):
    """
    If either lower or upper bounds are specified, turn into a single length
    two numpy array of [lower,upper]. If either was not specified, or was a 
    nan in the spreadsheet, convert to an appropriate infinity. 

    Parameters
    ----------
    out : dict
        dictionary of parameter values

    Returns
    -------
    out : dict
        dictionary of parameter values with bounds cleaned up
    """
    
    for p in out:

        # Initialize bounds 
        set_bounds = False
        bounds = [-np.inf,np.inf]

        # If we see a lower bound, grab it
        if "lower_bound" in out[p]:
            set_bounds = True
            bounds[0] = check_float(value=out[p].pop("lower_bound"),
                                    variable_name=f"param {p} lower_bound",
                                    allow_nan=True)
            
        # If we see an upper bound, grab it
        if "upper_bound" in out[p]:
            set_bounds = True
            bounds[1] = check_float(value=out[p].pop("upper_bound"),
                                    variable_name=f"param {p} upper_bound",
                                    allow_nan=True)
            
        # If either bound is nan, set to infinity
        if np.isnan(bounds[0]):
            bounds[0] = -np.inf
        if np.isnan(bounds[1]):
            bounds[1] = np.inf

        # If we saw a bound, record it
        if set_bounds:
            out[p]["bounds"] = np.array(bounds)

    return out

def _cleanup_priors(out):
    """
    If either prior_mean and prior_std are specified, turn into a single length
    two numpy array of [mean,std]. If either was not specified, throw an error. 

    Parameters
    ----------
    out : dict
        dictionary of parameter values

    Returns
    -------
    out : dict
        dictionary of parameter values with bounds cleaned up
    """

    for p in out:
    
        prior_mean = None
        prior_std = None

        # Try to load prior_mean. If nan, make None
        if "prior_mean" in out[p]:
            prior_mean = check_float(out[p].pop("prior_mean"),
                                     variable_name=f"param {p} prior_mean",
                                     allow_nan=True)
                                     
        # Try to load prior_std. If nan, make None
        if "prior_std" in out[p]:
            prior_std = check_float(out[p].pop("prior_std"),
                                    variable_name=f"param {p} prior_std",
                                    allow_nan=True)

        # neither specified with non-nan value. Continue
        if prior_mean is None and prior_std is None:
            continue

        # If have one non-nan value, throw an error
        if prior_mean is None or prior_std is None:
            err = "if either prior_mean or prior_std is specified, both must be specified\n"
            raise ValueError(err)

        if not np.isnan(prior_mean) and np.isnan(prior_std):
            err = "if either prior_mean or prior_std is specified, both must be specified\n"
            raise ValueError(err)
        
        if np.isnan(prior_mean) and not np.isnan(prior_std):
            err = "if either prior_mean or prior_std is specified, both must be specified\n"
            raise ValueError(err)


        # Record output
        out[p]["prior"] = np.array([prior_mean,prior_std])

    return out

def load_param_spreadsheet(spreadsheet):
    """
    Load information about fit parameters from a spreadsheet. The 'param' 
    column is required. Function will read the 'guess', 'prior_mean',
    'prior_std', 'lower_bound', 'upper_bound', and 'fixed' columns. All other
    columns are ignored. The 'param' column is read as strings. All entries must
    be unique. The 'fixed' column is read as bools. All other columns are read
    as floats. The code does some validation of these inputs. 
    
    Parameters
    ----------
    spreadsheet : str or pandas.DataFrame
        spreadsheet to read data from. If a string, the program will treat the 
        input as a filename and attempt to read (xslx, csv, tsv, and txt) will
        be recognized. If a dataframe, this will be treated as is. 
    
    Returns
    -------
    out : dictionary
        dictionary keying parameter names to properties (guess, prior_mean, 
        prior_std, lower_bound, upper_bound, and fixed) that key to the values
        of those inputs.
    """

    # read spreadsheet
    df = read_spreadsheet(spreadsheet=spreadsheet)

    # Make sure 'param' is present
    if "param" not in df.columns:
        err = "param must be a column in the spreadsheet\n"
        raise ValueError(err)
    
    # Make sure 'param' are all unique
    params = [str(p).strip() for p in df["param"]]
    if len(params) != len(np.unique(params)):
        err = "all entries in 'param' must be unique\n"
        raise ValueError(err)
    
    # Grab columns in dataframe
    columns_in_df = set(df.columns)

    # Columns we might want to grab
    columns_to_look_for = set(["guess","prior_mean","prior_std",
                               "lower_bound","upper_bound","fixed"])
    
    # Find columns in dataframe we want
    recognized_columns = list(columns_to_look_for.intersection(columns_in_df))

    # If no columns were recognized...
    if len(recognized_columns) == 0:
        err = "no recognized columns in the spreadsheet.\n"
        raise ValueError(err)

    # Go through each parameter...
    out = {}
    for i in range(len(df.index)):
        p = params[i]
        out[p] = {}

        # Go through each column of interest...
        for c in recognized_columns:

            # And record value
            out[p][c] = df.loc[df.index[i],c]

    # Clean up resulting inputs
    out = _cleanup_guess(out)
    out = _cleanup_fixed(out)
    out = _cleanup_bounds(out)
    out = _cleanup_priors(out)

    return out
        


    
