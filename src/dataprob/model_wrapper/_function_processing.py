"""
Functions for processing function signatures and identifying fit parameters.
"""

import inspect

# Various codes classifying argument types
KWARGS_KIND = inspect.Parameter.VAR_KEYWORD
ARGS_KIND = inspect.Parameter.VAR_POSITIONAL
EMPTY_DEFAULT = inspect.Parameter.empty

def analyze_fcn_sig(fcn):
    """
    Extract information about function being fit.

    Parameters
    ----------
    fcn : callable
        function or method used for fitting

    Returns
    -------
    all_args : list
        list of string names for all function arguments. This excludes
        *args and **kwargs
    can_be_fit : dict
        dictionary of arguments that can concievably fit. parameter names
        are keys, values are parameter defaults (float or None)
    cannot_be_fit : dict
        dictionary of arguments that cannot be fit. parameter names are
        keys, values are parameter defaults
    has_kwargs : bool
        whether or not this function takes kwargs
    """

    # Get function signature
    sig = inspect.signature(fcn)

    # Function outputs
    all_args = []
    can_be_fit = {}
    cannot_be_fit = {}
    has_kwargs = False

    # Go through parameters
    for p in sig.parameters:

        # If kwargs, record we saw it and skip
        if sig.parameters[p].kind is KWARGS_KIND:
            has_kwargs = True
            continue

        # If args, skip
        if sig.parameters[p].kind is ARGS_KIND:
            continue

        all_args.append(p)

        # Get default for argument
        default = sig.parameters[p].default

        # If this is iterable, assume it is not fittable. (Putting this here
        # prevents a bad comparison to multiple values in the next comparison to
        # EMPTY_DEFAULT)
        if hasattr(default,"__iter__"):
            cannot_be_fit[p] = default
            continue

        # If empty, assume it is fittable
        if default == EMPTY_DEFAULT:
            can_be_fit[p] = None    
            continue

        # Fittable if it can be coerced as a float
        try:
            can_be_fit[p] = float(default)
        except (TypeError,ValueError):
            cannot_be_fit[p] = default

    return all_args, can_be_fit, cannot_be_fit, has_kwargs


def reconcile_fittable(fittable_params,
                       all_args,
                       can_be_fit,
                       cannot_be_fit,
                       has_kwargs):
    """
    Find fittable and not fittable parameters for this function. 

    Parameters
    ----------
    fittable_params : list-like or None
        list of parameter names to fits (strings). If None, infer the
        fittable parameters
    all_args : list
        list of string names for all function arguments. This excludes
        *args and **kwargs
    can_be_fit : dict
        dictionary of arguments that can concievably fit. parameter names
        are keys, values are parameter defaults (float or None)
    cannot_be_fit : dict
        dictionary of arguments that cannot be fit. parameter names are
        keys, values are parameter defaults
    has_kwargs : bool
        whether or not this function takes kwargs

    Returns
    -------
    fittable_params : list-like
        list of fittable parameters built from fittable_params input and
        can_be_fit
    not_fittable_params : list-like
        list of unfittable params built from all_args and cannot_be_fit
    """
    
    # If fittable_params are not specified, construct
    if fittable_params is None:

        fittable_params = []
        for a in all_args:
            if a in can_be_fit:
                fittable_params.append(a)
            else:
                break
            
    if len(fittable_params) == 0:
        err = "no parameters to fit!\n"
        raise ValueError(err)

    for p in fittable_params:
        
        if p in cannot_be_fit:
            err = f"parameter '{p}' cannot be fit. It should have an empty\n"
            err += f"or float default argument in the function definition.\n"
            raise ValueError(err)

        if p not in can_be_fit and not has_kwargs:
            err = f"parameter '{p}' cannot be fit because is not in the\n"
            err += f"function definition.\n"
            raise ValueError(err)
        
    not_fittable_params = []
    for p in all_args:
        if p not in fittable_params:
            not_fittable_params.append(p)

    return fittable_params, not_fittable_params



def param_sanity_check(fittable_params,
                       reserved_params=None):
    """
    Check fittable parameters against list of reserved parameters.

    Parameters
    ----------
    fittable_params : list
        list of parameters to fit
    reserved_params : list
        list of reserved names we cannot use for parameters
    """

    if reserved_params is None:
        reserved_params = []

    for p in fittable_params:
        if p in reserved_params:
            err = f"parameter '{p}' is reserved by dataprob. Please use a different parameter name\n"
            raise ValueError(err)

    return fittable_params

def analyze_vector_input_fcn(fcn):
    """
    Extract information about function being fit.

    Parameters
    ----------
    fcn : callable
        function or method used for fitting

    Returns
    -------
    first_arg : str
        name of first argument
    other_kwargs  : dict
        dictionary keying all remaining arguments to their default values. 
        arguments with no default are assigned a default of None. 
    """

    sig = inspect.signature(fcn)

    first_arg = None
    other_kwargs = {}
    for p in sig.parameters:

        # Skip kwargs and args
        if sig.parameters[p].kind in [KWARGS_KIND,ARGS_KIND]:
            continue
        
        if first_arg is None:
            first_arg = p
            continue

        default = sig.parameters[p].default
        if default == EMPTY_DEFAULT:
            default = None

        other_kwargs[p] = default

    return first_arg, other_kwargs

        
        




