
from dataprob.model_wrapper.model_wrapper import ModelWrapper
from dataprob.model_wrapper.vector_model_wrapper import VectorModelWrapper
from dataprob.model_wrapper.read_spreadsheet import read_spreadsheet

from dataprob.check import check_bool

import pandas as pd

def wrap_function(some_function,
                  fit_parameters=None,
                  vector_first_arg=False):
    """
    Wrap a function for regression or Bayesian sampling. 

    Parameters
    ----------
    some_function : callable
        A function that takes at least one argument and returns a float value 
        or float numpy array. Fitter objects will compare the outputs of this 
        function against y_obs. 
    fit_parameters : list, dict, str, pandas.DataFrame; optional
        fit_parameters lets the user specify information about the parameters 
        in the fit. See Note below for details.
    vector_first_arg : bool, default=False
        If True, the first argument of the function is taken as a vector of 
        parameters to fit. All other arguments to some_function are treated as 
        non-fittable parameters. Fit_parameters must then specify the names of
        each vector element. 

    Returns
    -------
    mw : ModelWrapper
        ModelWrapper instance can be fed directly into a Fitter.fit method. The
        user can also manipulate fit parameters prior to the analysis. 

    Note
    ----
    There are two classes of parameters to each model. Fittable parameters are
    visible to Fitter instances (such as the ML fitter or Bayesian sampler) and
    are thus regressed/sampled. Non-fittable parameters are fixed and passed
    into `some_function` whenever it is called, but are invisible to the Fitter. 

    The software uses the signature of `some_function`, `fit_parameters`, and
    `vector_first_arg` to figure out what fit parameters to use. 
    
    In the simplest case (`fit_parameters is None`, `vector_first_arg is False`),
    the software will infer the fittable and non-fittable parameters from the
    `some_function` signature. It will grab the first N arguments with no
    default or whose default can be coerced to a float. The remaining arguments
    are treated as non-fittable parameters. Consider the example:

        `some_function == my_func(a,b=1,c="test",d=1)`

    The software will find the fittable parameters `a` and `b`, setting the
    guesses to `a = 0` and `b = 1`. The `c` and `d` parameters will be set as
    non-fittable.  
    
    If fittable_parameters is defined, it can override this default. For 
    example, if `fit_parameters = ['a','d']`, `a` and `d` will be fittable
    parameters and `b` and `c` will be non-fittable parameters. Except for two
    special cases described below, the parameters in `fit_parameters` must match
    the parameters in the function signature. The parameters `a`, `b`, and `d` 
    can be specified as fittable; the parameter `c` cannot because its default
    argument is a string. 

    NOTE: `fit_parameters` is treated as an exhaustive list of fittable 
    parameters. If specified, *only* the parameters in the list will be
    fittable.

    `fit_parameters` can differ from the parameters in the signature of 
    `some_function` in two cases: 
    
    1)  If the signature of `some_function` contains `**kwargs`, `fit_parameters`
        can be used to specify parameters to pass into some_function that are
        note explicitly delineated in the function signature. For example:

            `some_function == my_func(a,**kwargs)`
    
        would allow `fit_parameters = ['a','b','c']`. The `b` and `c` parameters 
        would be passed in as keyword arguments. (The code does not check 
        whether `my_func` can take those keyword arguments; that is the user's
        responsibility) 

    2)  If `vector_first_arg` is `True`, `fit_parameters` defines the parameters
        to pass in as a numpy.ndarray as the first function argument. If
        `vector_first_arg` is `True`, `fit_parameters` is required. All 
        function arguments besides this vector are treated as non-fittable 
        parameters. 
    
    Finally, `fit_parameters` can be used to pass in other information about 
    the fit parameters. This includes the parameter guess, whether or not it is
    fixed during the regression, its bounds, and the mean and standard deviation
    of a gaussian prior to apply to that fit parameter (Bayesian sampling only).
    This information can either be passed in via a dictionary or dataframe. 
        
    If `fit_parameters` comes in as a dictionary, the keys should be the
    parameter names (just like the entries to a `fit_parameters` list). The
    values should be dictionaries keying parameter attributes to their values.
    For example:

        `fit_parameters = {"K":{"guess":1,"bounds":(-np.inf,0)}}`
    
    would indicate that parameter "K" should have a guess of 1 and bounds from
    negative infinity to zero. 
    
    The allowed keys are: 

    |----------+--------------------------------------------------------------------------|
    | key      | value                                                                    |
    |----------+--------------------------------------------------------------------------|
    | 'guess'  | single float value (must be within bounds, if specified)                 |
    | 'fixed'  | True of False                                                            | 
    | 'bounds' | (lower,upper) as floats (-np.inf,np.inf) allowed                         | 
    | 'prior'  | (mean,stdev) as floats (np.nan,np.nan) allowed, meaning uniform prior    |
    |----------+--------------------------------------------------------------------------| 

    If `fit_parameters` comes in as a dataframe, the dataframe can have the
    following columns. 

    |---------------+---------------------------------------------------------------------|
    | key           | value                                                               |
    |---------------+---------------------------------------------------------------------|
    | 'param'       | string name of the parameter                                        |
    | 'guess'       | guess as single float value (must be within bounds, if specified)   |
    | 'fixed'       | True of False                                                       | 
    | 'lower_bound' | single float value; -np.inf allowed                                 | 
    | 'upper_bound' | single float value; np.inf allowed                                  | 
    | 'prior_mean'  | single float value; np.nan allowed                                  |
    | 'prior_std'   | single float value; np.nan allowed                                  |
    |---------------+---------------------------------------------------------------------| 

    If `fit_parameters` comes in as a string, this function will treat it as 
    the name of a spreadsheet file to read into a dataframe.
    """
    
    vector_first_arg = check_bool(value=vector_first_arg,
                                  variable_name="vector_first_arg")
    
    # Select the appropriate ModelWrapper instance to use
    if vector_first_arg:
        mw_class = VectorModelWrapper
    else:
        mw_class = ModelWrapper
    
    # Figure out how to set up the ModelWrapper based on the type of 
    # fit_parameters
    fit_param_type = type(fit_parameters)

    # None --> not specified. Use ModelWrapper default scheme
    if issubclass(fit_param_type,type(None)):
        
        fit_param_list = None
        mw = mw_class(model_to_fit=some_function,
                      fittable_params=fit_param_list)

    # List --> send in a list of fit parameters
    elif issubclass(fit_param_type,list):

        fit_param_list = fit_parameters
        mw = mw_class(model_to_fit=some_function,
                      fittable_params=fit_param_list)

    # dict --> send in keys as a list of fit parameters, then load the parameter
    # values in via the load_param_dict method. 
    elif issubclass(fit_param_type,dict):

        fit_param_list = list(fit_parameters.keys())
        mw = mw_class(model_to_fit=some_function,
                      fittable_params=fit_param_list)
        mw.load_param_dict(fit_parameters)

    # pd.DataFrame or str: treat as a spreadsheet. 
    elif issubclass(fit_param_type,pd.DataFrame) or issubclass(fit_param_type,str):

        # Read fit_parameters spreadsheet (or get copy of dataframe)
        fit_parameters = read_spreadsheet(fit_parameters)
    
        # Get list of fit parameters
        if "param" not in fit_parameters.columns:
            err = "fit_parameters DataFrame must have a 'param' column\n"
            raise ValueError(err)
        fit_param_list = list(fit_parameters["param"])

        # Initialize class, then load fit parameter data from the spreadsheet
        mw = mw_class(model_to_fit=some_function,
                      fittable_params=fit_param_list)
        mw.load_param_spreadsheet(fit_parameters)

    else:
    
        err = "fit_parameters not recognized. If specified, fit_parameters\n"
        err += "must be a list, dictionary, pandas DataFrame, or filename\n"
        err += "pointing to a spreadsheet. See the wrap_model docstring\n"
        err += "for details.\n"
        raise ValueError(err)
    
    return mw

        
        