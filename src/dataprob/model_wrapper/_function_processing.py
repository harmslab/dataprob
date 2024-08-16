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
        whether or not this function takes **kwargs
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
                       not_fittable_params,
                       all_args,
                       can_be_fit,
                       cannot_be_fit,
                       has_kwargs):
    """
    Find fittable and not fittable parameters for this function. 

    Parameters
    ----------
    fittable_params : list-like or None
        list of fittable parameter names. if None, infer
    not_fittable_params : list-like or None
        list of non-fittable parameter names. if None, infer. 
    all_args : list
        list of string names for all function arguments. This excludes
        *args and **kwargs
    can_be_fit : dict
        dictionary of arguments that can conceivably fit. parameter names
        are keys, values are parameter defaults (float or None)
    cannot_be_fit : dict
        dictionary of arguments that cannot be fit. parameter names are
        keys, values are parameter defaults
    has_kwargs : bool
        whether or not this function takes **kwargs

    Returns
    -------
    fittable_params : list-like
        list of fittable parameters built from fittable_params input and
        can_be_fit
    not_fittable_params : list-like
        list of unfittable params built from nonfittable_params, all_args, and
        cannot_be_fit
    """
    
    # Make sure the user didn't send in the same parameter as both a fittable
    # and non-fittable parameter
    if fittable_params is not None and not_fittable_params is not None:
        fittable_set = set(fittable_params)
        not_fittable_set = set(not_fittable_params)
        intersect = fittable_set.intersection(not_fittable_set)
        if len(intersect) != 0:
            err = "a parameter cannot be in fittable_params and not fittable_params.\n"
            err += f"Bad parameters: {str(intersect)}\n"
            raise ValueError(err)
            
    # If fittable_params are not specified, construct
    if fittable_params is None:

        fittable_params = []
        for a in all_args:
            if a in can_be_fit:
                fittable_params.append(a)
            else:
                break

    # Go through all fittable parameters
    for p in fittable_params:
        
        # Not fittable based on sig -- die
        if p in cannot_be_fit:
            err = f"parameter '{p}' cannot be fit. It should have an empty\n"
            err += f"or float default argument in the function definition.\n"
            raise ValueError(err)

        # Not found, and no **kwargs around. 
        if p not in can_be_fit and not has_kwargs:
            err = f"fittable parameter '{p}' is not in the function definition\n"
            raise ValueError(err)

    if not_fittable_params is None:
        not_fittable_params = []

    # Go through all nonfittable params
    for p in not_fittable_params:

        # Parameter not in function signature and the signature does not have 
        # **kwargs. Fail. 
        in_can_fit = p in can_be_fit
        in_cannot_fit = p in cannot_be_fit
        if not in_can_fit and not in_cannot_fit and not has_kwargs:
            err = f"not_fittable parameter '{p}' is not in the function definition\n"
            raise ValueError(err)

    # Filter kwargs to remove any user-specified nonfittable_params
    fittable_params = [p for p in fittable_params
                       if p not in not_fittable_params]

    # If we get here and do not have a fittable parameter, bad news. 
    if len(fittable_params) == 0:
        err = "no parameters to fit!\n"
        raise ValueError(err)
    
    # Make a final list of not_fittable_params -- everything that is not 
    # fittable. 
    final_not_fittable = []
    for p in all_args:
        if p not in fittable_params:
            final_not_fittable.append(p)

    # Grab anything sent in by the user that is not already in the definition. 
    # (Do this way so args are in order from signature, then in order sent in by
    # user for any **kwargs. 
    for p in not_fittable_params:
        if p not in final_not_fittable:
            final_not_fittable.append(p)

    return fittable_params, final_not_fittable


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
    has_kwargs : bool
        whether or not this function takes **kwargs
    """

    sig = inspect.signature(fcn)

    first_arg = None
    other_kwargs = {}
    has_kwargs = False
    for p in sig.parameters:

        if sig.parameters[p].kind == KWARGS_KIND:
            has_kwargs = True
            continue

        # Skip kwargs and args
        if sig.parameters[p].kind == ARGS_KIND:
            continue
        
        if first_arg is None:
            first_arg = p
            continue

        default = sig.parameters[p].default
        if default == EMPTY_DEFAULT:
            default = None

        other_kwargs[p] = default

    return first_arg, other_kwargs, has_kwargs

        
        




