"""
Functions for processing parameter dataframes.
"""

import numpy as np
import pandas as pd

def _check_name(param_df,param_in_order):
    """
    Check name column for sanity, set it as the index, and order dataframe
    according to param_in_order
    """

    if "name" not in param_df.columns:
        err = "param_df must have a name column\n"
        raise ValueError(err)
    
    if len(set(param_in_order)) != len(param_in_order):
        err = "param_in_order must all have unique parameters\n"
        raise ValueError(err)

    # work on a copy
    param_df = param_df.copy()

    # Coerce to string
    param_df["name"] = param_df["name"].astype(str)

    names_in_df = set(param_df["name"])
    names_in_mw = set(param_in_order)

    if names_in_df != names_in_mw:
        err = "\nValues in the 'name' column in a parameter dataframe must\n"
        err += "be identical to the fit parameter names.\n\n"
        
        missing_values = names_in_mw - names_in_df
        if len(missing_values) > 0:
            err += "Missing values:\n"
            for v in missing_values:
                err += f"    {v}\n"
            err += "\n"

        extra_values = names_in_df - names_in_mw
        if len(extra_values) > 0:
            err += "Extra values:\n"
            for v in extra_values:
                err += f"    {v}\n"
            err += "\n"

        raise ValueError(err)
    
    # Make sure the index is the parameter name
    param_df.index = param_df["name"]
    param_df = param_df.loc[param_in_order,:]

    return param_df

def _build_columns(param_df,default_guess):
    """
    Build any missing columns and coerce them to the correct types.
    """

    # ----------------------------------------------------------------------
    # Build any missing columns

    if "guess" not in param_df.columns:
        param_df["guess"] = default_guess
    if "fixed" not in param_df.columns:
        param_df["fixed"] = False
    if "lower_bound" not in param_df.columns:
        param_df["lower_bound"] = -np.inf
    if "upper_bound" not in param_df.columns:
        param_df["upper_bound"] = np.inf
    if "prior_mean" not in param_df.columns:
        param_df["prior_mean"] = np.nan
    if "prior_std" not in param_df.columns:
        param_df["prior_std"] = np.nan

    # ----------------------------------------------------------------------
    # Coerce column types

    float_columns = ["guess",
                     "lower_bound","upper_bound",
                     "prior_mean","prior_std"]

    for fc in float_columns:

        # start with the pandas caster as this is smart and robust
        try:
            param_df[fc] = pd.to_numeric(param_df[fc])
        except Exception as e:
            err = f"Could not coerce all entries in the '{fc}' column to float\n"
            raise ValueError(err) from e
        
        # then do a direct cast to float
        param_df[fc] = param_df[fc].astype(float)
    
    bool_columns = ["fixed"]

    for bc in bool_columns:

        try:
            param_df[bc] = param_df[bc].astype(bool)
        except Exception as e:
            err = f"Could not coerce all entries in the '{bc}' column to bool\n"
            raise ValueError(err) from e

    return param_df

def _check_bounds(param_df):
    """
    Check upper_bound and lower_bound columns for sanity.
    """

    # Set nan values to be -np.inf and +np.inf
    lower_bound_nan = pd.isna(param_df["lower_bound"])
    upper_bound_nan = pd.isna(param_df["upper_bound"])
    param_df.loc[lower_bound_nan,"lower_bound"] = -np.inf
    param_df.loc[upper_bound_nan,"upper_bound"] = np.inf

    # Check for bounds that are inconsistent.
    bad_bound_mask = param_df["lower_bound"] >= param_df["upper_bound"]

    if np.sum(bad_bound_mask) > 0:
        
        bad_df = param_df.loc[bad_bound_mask,["name",
                                              "lower_bound","upper_bound"]]
        
        err = "\nBounds must have lower_bound < upper_bound. -np.inf is\n"
        err += "allowed lower_bound; np.inf is allowed for upper_bound.\n"
    
        err += "\nBad parameters are:\n"
        err += f"\n{repr(bad_df)}\n\n"

        raise ValueError(err)

    return param_df

def _check_guesses(param_df):
    """
    Check guess column for sanity.
    """
    
    guesses = param_df["guess"]
    too_low_mask = guesses < param_df["lower_bound"]
    too_high_mask = guesses > param_df["upper_bound"]
    is_nan_mask = np.isnan(guesses)
    bad_guesses = np.logical_or.reduce((too_low_mask,
                                        too_high_mask,
                                        is_nan_mask)) 
    if np.sum(bad_guesses) > 0:

        bad_df = param_df.loc[bad_guesses,["name","guess",
                                           "lower_bound","upper_bound"]]

        err = "\nGuess values must be non-nan and between lower_bound and\n"
        err += "upper_bound.\n"

        err += "\nBad parameters are:\n"
        err += f"\n{repr(bad_df)}\n\n"

        raise ValueError(err)
    
    return param_df
            
def _check_priors(param_df):
    """
    Check prior_mean and prior_std columns for sanity.
    """

    # Look for partially defined parameters using a XOR gate, which is true
    # if mean OR std is nan, but false if neither is nan or both are nan. 
    prior_mean_nan = pd.isna(param_df["prior_mean"])
    prior_std_nan = pd.isna(param_df["prior_std"])
    nan_xor = np.logical_or(
        np.logical_and(prior_mean_nan,
                        np.logical_not(prior_std_nan)),
        np.logical_and(np.logical_not(prior_mean_nan),
                        prior_std_nan)
    )
    if np.sum(nan_xor) > 0:

        bad_df = param_df.loc[nan_xor,["name","prior_mean","prior_std"]]

        err = "\nIf either prior_mean and prior_std are defined for a\n"
        err += "given parameter, both must be non-nan. If prior_mean\n"
        err += "and prior_std are both set to np.nan, the parameter uses\n"
        err += "uniform priors within the specified bounds.\n"
        
        err += "\nBad parameters are:\n"
        err += f"\n{repr(bad_df)}\n\n"

        raise ValueError(err)
    
    # Look for prior_std that are defined and <= 0. 
    bad_prior_std = param_df["prior_std"] <= 0
    if np.sum(bad_prior_std) > 0:

        bad_df = param_df.loc[bad_prior_std,["name","prior_std"]]

        err = "\nIf defined, prior_std must be > 0.\n"

        err += "\nBad parameters are:\n"
        err += f"\n{repr(bad_df)}\n\n"
        
        raise ValueError(err)

    # Look for infinite priors
    prior_mean_inf = np.isinf(param_df["prior_mean"])
    prior_std_inf = np.isinf(param_df["prior_std"])
    bad_prior_inf = np.logical_or(prior_mean_inf,prior_std_inf)
    if np.sum(bad_prior_inf) > 0:
        
        bad_df = param_df.loc[bad_prior_inf,["name","prior_mean","prior_std"]]

        err = "\nprior_mean and prior_std must be finite. If prior_mean and\n"
        err += "prior_std are set to np.nan, the parameter uses uniform\n"
        err += "priors within the specified bounds.\n"

        err += "\nBad parameters are:\n"
        err += f"\n{repr(bad_df)}\n\n"

        raise ValueError(err)

    return param_df

def _df_to_dict(df):
    """
    Convert a dataframe into a nested dictionary (out_dict[name][column]). This
    is useful in case the input dataframe has only partial coverage of a larger
    dataframe. 
    """

    if "name" not in df.columns:
        err = "dataframe must have a name column\n"
        raise ValueError(err)
    
    # work on copy
    df = df.copy()

    # Coerce to string
    df["name"] = df["name"].astype(str)

    # make sure names are all unique
    if len(set(df["name"])) != len(df["name"]):
        err = "all 'name' entries in dataframe must be unique\n"
        raise ValueError(err)

    df.index = df["name"]

    out_dict = {}
    for p in df["name"]:
        out_dict[p] = {}
        for c in df.columns:
            out_dict[p][c] = df.loc[p,c]

    return out_dict


def validate_dataframe(param_df,
                       param_in_order,
                       default_guess=0):
    """
    Validate a parameter dataframe, returning it in a standardized format. This
    validates and possibly creates the core columns (name, guess, fixed,
    lower_bound, upper_bound, prior_mean, and prior_std). It also orders the 
    parameters according to param_in_order and sets the dataframe index to be
    the parameter name. Other columns are left intact, but ignored. 
    
    Parameters
    ----------
    param_df : pandas.DataFrame
        parameter dataframe to check
    param_in_order : list-like
        list of parameter names in the desired order
    default_guess : float, default = 0
        assign missing guess entries this value
        
    Returns
    -------
    param_df : pandas.DataFrame
        validated dataframe
    """
    
    # make sure the input is a dataframe
    if not issubclass(type(param_df),pd.DataFrame):
        err = "\nparam_df should be a pandas DataFrame\n"
        raise ValueError(err)

    # Check dataframe entries
    param_df = _check_name(param_df=param_df,
                           param_in_order=param_in_order)
    
    param_df = _build_columns(param_df=param_df,
                              default_guess=default_guess)
    
    param_df = _check_bounds(param_df=param_df)

    param_df = _check_guesses(param_df=param_df)

    param_df = _check_priors(param_df=param_df)


    return param_df


def param_into_existing(param_input,
                        param_df):
    """
    Load parameter information from param_input into param_df. 

    Parameters
    ----------
    param_input : dict or pandas.DataFrame
        this can be nested dictionary that keys parameter names to columns to
        values (param_input[name][column] -> value). it can also be a pandas
        dataframe with (minimally) a 'name' column. 
    param_df : pandas.DataFrame
        parameter dataframe into which we are loading data. 
        
    Notes
    -----
    Any information in param_df that is not explicitly defined in param_input
    will be left intact in param_df. For example, if param_df has entries for
    parameters 'K1' and 'K2', param_input could set values for 'K1' without
    altering 'K2'. This can happen at the attribute level as well. We could
    set the guess of 'K1' without altering the 'upper_bound'. 

    0. If a param_input is a dataframe, it must have a 'name' column that
       corresponds to the parameters in param_df. 
    1. The parameters in param_input must already be in param_df.
    2. Not all parameters in param_df must be in param_input. Parameters that 
       are not in param_input are not changed in param_df.  
    3. param_input can have columns that are not present in param_df. These 
       will be added to param_df. 
    """
    
    # If the param_input is a dataframe, convert to a dictionary
    if issubclass(type(param_input),pd.DataFrame):
        param_input = _df_to_dict(param_input)

    # If param_input is not a dictionary at this point, throw an error
    if not issubclass(type(param_input),dict):
        err = "param_input should be a pandas dataframe or dictionary\n"
        raise ValueError(err)
    
    # work on copy of parameter dataframe
    param_df = param_df.copy()

    # Go through parameters...
    added_columns = []
    for p in param_input:
        
        # Make sure parameter is in model
        if p not in param_df["name"]:
            err = f"parameter {p} is not a parameter in the model\n"
            raise ValueError(err)
        
        if not issubclass(type(param_input[p]),dict):
            err = f"\nparam_input['{p}'] should be a dictionary\n"
            err += f"\n{param_into_existing.__doc__}\n\n"
            raise ValueError(err)

        # Go through input columns for parameter
        for c in param_input[p]:

            # Decide if we need to add a new column
            if c not in param_df.columns:
                added_columns.append(c)
                continue
            
            # If column already exists, add to it
            param_df.loc[p,c] = param_input[p][c]
    
    # Build columns that were not in param_df
    for c in added_columns:
        new_column = []
        for p in param_df["name"]:

            # Parameter not in input
            if p not in param_input:
                new_column.append(None)
                continue
        
            # Parameter in input, but not column
            if c not in param_input[p]:
                new_column.append(None)
                continue

            # Parameter and column in input
            new_column.append(param_input[p][c])
            
        param_df[c] = new_column
    
    return param_df