"""
Functions to check/process `bool`, `float`, `int`, and iterable arguments in
functions.
"""

import numpy as np
import pandas as pd

def check_bool(value,variable_name=None):
    """
    Process a `bool` argument and do error checking.

    Parameters
    ----------
    value :
        input value to check/process
    variable_name : str
        name of variable (string, for error message)

    Returns
    -------
    bool
        validated/coerced bool

    Raises
    ------
    ValueError
        If value cannot be interpreted as a bool
    """

    if np.issubdtype(type(value),np.bool_):
        return bool(value)

    try:

        # See if this is an iterable
        if hasattr(value,"__iter__"):
            raise ValueError

        # See if this is a naked type
        if issubclass(type(value),type):
            raise ValueError

        # See if this is a float that is really, really close to an integer.
        if value != 0:
            if not np.isclose(round(value,0)/value,1):
                raise ValueError

        # Final check to make sure it's really a bool not an integer pretending
        # to be bool
        value = int(value)
        if value not in [0,1]:
            raise ValueError

        value = bool(value)

    except (TypeError,ValueError):

        if variable_name is not None:
            err = f"\n{variable_name} '{value}' must be True or False.\n\n"
        else:
            err = f"\n'{value}' must be True or False.\n\n"
        err += "\n\n"

        raise ValueError(err)

    return value


def check_float(value,
                variable_name=None,
                minimum_allowed=-np.inf,
                maximum_allowed=np.inf,
                minimum_inclusive=True,
                maximum_inclusive=True,
                allow_nan=False):
    """
    Process a `float` argument and do error checking.

    Parameters
    ----------
    value :
        input value to check/process
    variable_name : str
        name of variable (string, for error message)
    minimum_allowed : float, default=-np.inf
        minimum allowable value for the variable
    maximum_allowed : float, default=np.inf
        maximum allowable value for the variable
    minimum_inclusive : bool, default=True
        whether lower bound is inclusive
    maximum_inclusive : bool, default=True
        whether upper bound is inclusive
    allow_nan : bool, default=False
        allow nan (or pd.NA or None), which are all coerced to np.nan

    Returns
    -------
    float
        validated/coerced float

    Raises
    ------
    ValueError
        If value cannot be interpreted as a float
    """

        
    try:

        # Try to cast as string to an integer
        if issubclass(type(value),str):
            value = float(value)

        # See if this is an iterable
        if hasattr(value,"__iter__"):
            raise ValueError
        
        # See if this is a naked type
        if issubclass(type(value),type):
            raise ValueError
        
        # If this is nan, null or None... If we are allowing nan,
        # return nan and do not do further checks. Otherwise, throw an error
        if pd.isnull(value) or value is None:
            if allow_nan:
                return np.nan
            raise ValueError

        value = float(value)

        if minimum_inclusive:
            if value < minimum_allowed:
                raise ValueError
        else:
            if value <= minimum_allowed:
                raise ValueError

        if maximum_inclusive:
            if value > maximum_allowed:
                raise ValueError
        else:
            if value >= maximum_allowed:
                raise ValueError

    except (ValueError,TypeError):

        if minimum_inclusive:
            min_o = "<="
        else:
            min_o = "<"

        if maximum_inclusive:
            max_o = "<="
        else:
            max_o = "<"

        bounds = f"{minimum_allowed} {min_o} {variable_name} {max_o} {maximum_allowed}"

        if variable_name is not None:
            err = f"\n{variable_name} '{value}' must be a float:\n\n"
        else:
            err = f"\n'{value}' must be a float:\n\n"

        if not (minimum_allowed is None and maximum_allowed is None):
            err += bounds

        err += "\n\n"

        raise ValueError(err)

    return value

def check_int(value,
              variable_name=None,
              minimum_allowed=None,
              maximum_allowed=None,
              minimum_inclusive=True,
              maximum_inclusive=True):
    """
    Process an `int` argument and do error checking.

    Parameters
    ----------
    value :
        input value to check/process
    variable_name : str
        name of variable (string, for error message)
    minimum_allowed : float, optional
        minimum allowable value for the variable
    maximum_allowed : float, optional
        maximum allowable value for the variable
    minimum_inclusive : bool, default=True
        whether lower bound is inclusive
    maximum_inclusive : bool, default=True
        whether upper bound is inclusive

    Returns
    -------
    int
        validated/coerced integer

    Raises
    ------
    ValueError
        If value cannot be interpreted as a int
    """


    try:

        # Try to cast as string to an integer
        if issubclass(type(value),str):
            value = int(value)

        # See if this is an iterable
        if hasattr(value,"__iter__"):
            raise ValueError

        # See if this is a naked type
        if issubclass(type(value),type):
            raise ValueError

        # If this is a float to int cast, make sure it does not have decimal
        floored = np.floor(value)
        if value != floored:
            raise ValueError

        # Make int cast
        value = int(value)

        if minimum_allowed is not None:
            if minimum_inclusive:
                if value < minimum_allowed:
                    raise ValueError
            else:
                if value <= minimum_allowed:
                    raise ValueError

        if maximum_allowed is not None:
            if maximum_inclusive:
                if value > maximum_allowed:
                    raise ValueError
            else:
                if value >= maximum_allowed:
                    raise ValueError

    except (ValueError,TypeError,OverflowError):

        if minimum_inclusive:
            min_o = "<="
        else:
            min_o = "<"

        if maximum_inclusive:
            max_o = "<="
        else:
            max_o = "<"

        bounds = f"{minimum_allowed} {min_o} {variable_name} {max_o} {maximum_allowed}"

        if variable_name is not None:
            err = f"\n{variable_name} '{value}' must be an integer:\n\n"
        else:
            err = f"\n'{value}' must be an integer:\n\n"

        if not (minimum_allowed is None and maximum_allowed is None):
            err += bounds

        err += "\n\n"

        raise ValueError(err)

    return value

def check_array(value,
                variable_name=None,
                expected_shape=None,
                expected_shape_names=None,
                nan_allowed=True):

    """
    Do a sanity check on arguments that send in parameters (ln_like, etc.).

    Parameters
    ----------
    value :
        input value to check/process
    variable_name : str
        name of variable (string, for error message)
    expected_shape : tuple
        expected dimensions. None entries indicate that dimension must be 
        present but does not have a specified length
    expected_shape_names : str
        name of dimensions (string, for error message)
    name_allowed : bool, default=True
        if True, allow nan. If False, throw an error on nan
        
    Returns
    -------
    value : numpy.ndarray
        value validated and coerced (if necessary) into a float array
    """
    
    err = f"'{value}' must be a float numpy array"

    if variable_name is not None:
        err = f"{variable_name} {err}"
    
    if expected_shape is not None:
        if expected_shape_names is not None:
            err = f"{err} with shape {expected_shape_names}"
        else:
            err = f"{err} with shape {expected_shape}"
    
    err = f"\n{err}\n"

    if not hasattr(value,"__iter__"):
        raise ValueError(err)

    # Initial conversion to an object array, then filter pd.NA into np.nan. 
    # The initial conversion should eb extraordinarily robust. 
    value = np.array(value,dtype=object)
    value[pd.isna(value)] = np.nan
    
    # Coerce to float (could have np.nan)
    try:
        value = np.array(value,dtype=float)
    except Exception as e:
        err = f"{err} Could not coerce to a float numpy array\n"
        raise ValueError(err) from e
    
    # Check final shape
    if expected_shape is not None:

        if len(value.shape) != len(expected_shape):
            raise ValueError(err)
        
        for i in range(len(expected_shape)):
            if expected_shape[i] is not None:
                if value.shape[i] != expected_shape[i]:
                    raise ValueError(err)
    
    if not nan_allowed:
        num_nan = np.sum(np.isnan(value))
        if num_nan > 0:
            err = f"{err} without nan values. Array has {num_nan} nan entries.\n"
            raise ValueError(err)

    return value
