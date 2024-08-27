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

        # Fittable if it can be coerced as a float. Explicitly exclude bool
        try:
            if issubclass(type(default),bool):
                raise ValueError
            can_be_fit[p] = float(default)
        except (TypeError,ValueError):
            cannot_be_fit[p] = default

    return all_args, can_be_fit, cannot_be_fit, has_kwargs


def reconcile_fittable(fit_parameters,
                       non_fit_kwargs,
                       all_args,
                       can_be_fit,
                       cannot_be_fit,
                       has_kwargs):
    """
    Find fittable and not fittable parameters for this function. 

    Parameters
    ----------
    fit_parameters : list-like, optional
        list of fittable parameter names. if None, infer
    non_fit_kwargs : dict, optional
        non-fit keyword arguments to pass to the function. if None, infer. 
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
    fit_parameters : list
        list of fittable parameters built from fit_parameters input and
        can_be_fit
    not_fit_parameters : list
        list of unfittable params built from non_fit_kwargs, all_args, and
        cannot_be_fit
    """
    
    # Construct not_fit_parameters from non_fit_kwargs keys
    if non_fit_kwargs is not None:
        not_fit_parameters = list(non_fit_kwargs.keys())
    else:
        not_fit_parameters = []

    # Make sure the user didn't send in the same parameter as both a fittable
    # and non-fittable parameter
    if fit_parameters is not None:
        fittable_set = set(fit_parameters)
        not_fittable_set = set(not_fit_parameters)
        intersect = fittable_set.intersection(not_fittable_set)
        if len(intersect) != 0:
            err = "a parameter cannot be in fit_parameters and not fit_parameters.\n"
            err += f"Bad parameters: {str(intersect)}\n"
            raise ValueError(err)
            
    # If fit_parameters are not specified, construct
    if fit_parameters is None:
        fit_parameters = []
        for a in all_args:
            if a in can_be_fit:
                fit_parameters.append(a)
            else:
                break

    # Go through all fittable parameters
    for p in fit_parameters:
        
        # Not fittable based on sig -- die
        if p in cannot_be_fit:
            err = f"parameter '{p}' cannot be fit. It should have an empty\n"
            err += f"or float default argument in the function definition.\n"
            raise ValueError(err)

        # Not found, and no **kwargs around. 
        if p not in can_be_fit and not has_kwargs:
            err = f"fittable parameter '{p}' is not in the function definition\n"
            raise ValueError(err)

    # Go through all nonfittable params
    for p in not_fit_parameters:

        # Parameter not in function signature and the signature does not have 
        # **kwargs. Fail. 
        in_can_fit = p in can_be_fit
        in_cannot_fit = p in cannot_be_fit
        if not in_can_fit and not in_cannot_fit and not has_kwargs:
            err = f"not_fittable parameter '{p}' is not in the function definition\n"
            raise ValueError(err)

    # Filter kwargs to remove any user-specified nonfit_parameters
    fit_parameters = [p for p in fit_parameters
                       if p not in not_fit_parameters]

    # If we get here and do not have a fittable parameter, bad news. 
    if len(fit_parameters) == 0:
        err = "no parameters to fit!\n"
        raise ValueError(err)
    
    # Make a final list of not_fit_parameters -- everything that is not 
    # fittable. 
    final_not_fittable = []
    for p in all_args:
        if p not in fit_parameters:
            final_not_fittable.append(p)

    # Grab anything sent in by the user that is not already in the definition. 
    # (Do this way so args are in order from signature, then in order sent in by
    # user for any **kwargs. 
    for p in not_fit_parameters:
        if p not in final_not_fittable:
            final_not_fittable.append(p)

    return fit_parameters, final_not_fittable


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

        
        




