"""
Convenience function that initializes a ModelWrapper class from a user-
specified input function. 
"""

from dataprob.model_wrapper.model_wrapper import ModelWrapper
from dataprob.model_wrapper.vector_model_wrapper import VectorModelWrapper
from dataprob.util.read_spreadsheet import read_spreadsheet

from dataprob.util.check import check_bool

import pandas as pd

def wrap_function(some_function,
                  fit_parameters=None,
                  non_fit_kwargs=None,
                  vector_first_arg=False):
    """
    Wrap a function for regression or Bayesian sampling. 

    Parameters
    ----------
    some_function : callable
        A function that takes at least one argument and returns a float numpy
        array. Fitter objects will compare the outputs of this function against
        y_obs. 
    fit_parameters : list, dict, str, pandas.DataFrame; optional
        fit_parameters lets the user specify information about the parameters 
        in the fit. See Note below for details.
    non_fit_kwargs : dict
        non_fit_kwargs are keyword arguments for some_function that should not
        be fit but need to be specified to non-default values. 
    vector_first_arg : bool, default=False
        If True, the first argument of the function is taken as a vector of 
        parameters to fit. All other arguments to some_function are treated as 
        non-fittable parameters. fit_parameters must then specify the names of
        each vector element. 

    Returns
    -------
    mw : ModelWrapper
        ModelWrapper instance that can be used in a Fitter instance. 
    """
    
    vector_first_arg = check_bool(value=vector_first_arg,
                                  variable_name="vector_first_arg")
    
    # Select the appropriate ModelWrapper instance to use
    if vector_first_arg:
        mw_class = VectorModelWrapper
    else:
        mw_class = ModelWrapper
    
    # -------------------------------------------------------------------------
    # Figure out how to treat fit_parameters based on type

    fit_param_type = type(fit_parameters)

    if issubclass(fit_param_type,type(None)):
        fit_param_list = None
        fit_param_values = {}
    
    elif issubclass(fit_param_type,dict):
        fit_param_list = list(fit_parameters.keys())
        fit_param_values = fit_parameters
    
    elif issubclass(fit_param_type,pd.DataFrame) or issubclass(fit_param_type,str):

        # Read fit_parameters spreadsheet (or get copy of dataframe)
        fit_param_values = read_spreadsheet(fit_parameters)
        if "name" not in fit_param_values.columns:
            err = "fit_parameters DataFrame must have a 'name' column\n"
            raise ValueError(err)
        
        # Get list of parameters from the dataframe
        fit_param_list = list(fit_param_values["name"])

    elif hasattr(fit_param_type,"__iter__"):
        fit_param_list = fit_parameters
        fit_param_values = {}

    else:
        err = "fit_parameters not recognized. If specified, fit_parameters\n"
        err += "must be a list, dictionary, pandas DataFrame, or filename\n"
        err += "pointing to a spreadsheet. See the wrap_model docstring\n"
        err += "for details.\n"
        raise ValueError(err)

    
    # Create class with appropriate parameters
    mw = mw_class(model_to_fit=some_function,
                  fit_parameters=fit_param_list,
                  non_fit_kwargs=non_fit_kwargs)
    
    # Update fit parameters with values
    mw.update_params(fit_param_values)

    return mw

        
        