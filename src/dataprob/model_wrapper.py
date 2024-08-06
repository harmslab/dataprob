"""
Class for wrapping models for use in likelihood calculations. 
"""

from dataprob.fit_param import FitParameter

import numpy as np

import inspect

def _analyze_fcn_sig(fcn):
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

    # Various codes classifying argument types
    kwargs_kind = inspect.Parameter.VAR_KEYWORD
    args_kind = inspect.Parameter.VAR_POSITIONAL
    empty = inspect.Parameter.empty

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
        if sig.parameters[p].kind is kwargs_kind:
            has_kwargs = True
            continue

        # If args, skip
        if sig.parameters[p].kind is args_kind:
            continue

        all_args.append(p)

        # Get default for argument
        default = sig.parameters[p].default

        # If empty, assume it is fittable
        if default == empty:
            can_be_fit[p] = None    
            continue

        # Fittable if it can be coerced as a float
        try:
            can_be_fit[p] = float(default)
        except (TypeError,ValueError):
            cannot_be_fit[p] = default

    return all_args, can_be_fit, cannot_be_fit, has_kwargs

def _reconcile_fittable(fittable_params,
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


def _param_sanity_check(fittable_params,
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



class ModelWrapper:
    """
    Wrap a model for use in likelihood calculations.

    The first N arguments with no default argument or arguments whose
    default can be coerced as a float are converted to fit parameters. The
    remaining arguments are treated as non-fittable parameters.  A specific
    set of arguments to convert to fit parameters can be specified by
    specifying 'fittable_params'.
    """

    # Attributes to hold the fit parameters and other arguments to pass
    # to the model. These have to be defined across class because we are going
    # to hijack __getattr__ and __setattr__ and need to look inside this as soon
    # as we start setting attributes.
    _mw_fit_parameters = {}
    _mw_other_arguments = {}

    def __init__(self,model_to_fit,fittable_params=None):
        """

        Parameters
        ----------
        model_to_fit : callable
            a function or method to fit.
        fittable_params : list-like, optional
            list of arguments to fit.
        """

        # Define these here so __setattr__ and __getattr__ are looking at
        # instance-level attributes rather than class-level attributes.
        self._mw_fit_parameters = {}
        self._mw_other_arguments = {}

        self._model_to_fit = model_to_fit
        self._mw_load_model(fittable_params)

    def _mw_load_model(self,fittable_params):
        """
        Load a model into the wrapper, making the arguments into attributes.
        Fittable arguments are made into FitParameter instances.  Non-fittable
        arguments are set as generic attributes.

        Parameters
        ----------
        fittable_params : list-like or None
            list of parameters to fit 
        """

        all_args, can_be_fit, cannot_be_fit, has_kwargs = _analyze_fcn_sig(fcn=self._model_to_fit)

        fittable_params, not_fittable_parameters = \
            _reconcile_fittable(fittable_params=fittable_params,
                                all_args=all_args,
                                can_be_fit=can_be_fit,
                                cannot_be_fit=cannot_be_fit,
                                has_kwargs=has_kwargs)

        reserved_params = dir(self.__class__)
        fittable_params = _param_sanity_check(fittable_params=fittable_params,
                                              reserved_params=reserved_params)

        for p in fittable_params:

            if p in can_be_fit:
                guess = can_be_fit[p]
            else:
                guess = None
                
            self._mw_fit_parameters[p] = FitParameter(name=p,guess=guess)
        
        for p in not_fittable_parameters:
            
            if p in can_be_fit:
                starting_value = can_be_fit[p]
            else:
                starting_value = cannot_be_fit[p]
            self._mw_other_arguments[p] = starting_value
        
        self._update_parameter_map()

    def __setattr__(self, key, value):
        """
        Hijack __setattr__ so setting the value for fit parameters
        updates the fit guess.
        """

        # We're setting the guess of the fit parameter
        if key in self._mw_fit_parameters.keys():
            self._mw_fit_parameters[key].guess = value

        # We're setting another argument
        elif key in self._mw_other_arguments.keys():
            self._mw_other_arguments[key] = value

        elif key in ["bounds","guesses","names","fixed"]:
            err = f"'{key}' can only be set at the individual parameter level\n"
            err += "for a ModelWrapper instance.\n"
            raise TypeError(err)

        # Otherwise, just set it like normal
        else:
            super(ModelWrapper, self).__setattr__(key, value)

    def __getattr__(self,key):
        """
        Define __getattr__ to we get fit parameters and other arguments
        appropriately.
        """

        # We're getting a fit parameter
        if key in self._mw_fit_parameters.keys():
            return self._mw_fit_parameters[key]

        # We're getting another argument
        if key in self._mw_other_arguments.keys():
            return self._mw_other_arguments[key]

        # Otherwise, get like normal
        else:
            super(ModelWrapper,self).__getattribute__(key)

    def _update_parameter_map(self):
        """
        Update the map between the parameter vector that will be passed in to
        the fitter and the parameters in this wrapper. This
        """

        self._position_to_param = []
        self._mw_kwargs = {}
        for p in self._mw_fit_parameters.keys():
            if self._mw_fit_parameters[p].fixed:
                self._mw_kwargs[p] = self._mw_fit_parameters[p].value
            else:
                self._mw_kwargs[p] = None
                self._position_to_param.append(p)

        self._mw_kwargs.update(self._mw_other_arguments)

    def _mw_observable(self,params=None):
        """
        Actual function called by the fitter.
        """

        # If parameters are not passed, stick in the current parameter
        # values
        if params is None:
            for p in self.position_to_param:
                self._mw_kwargs[p] = self.fit_parameters[p].value
        else:
            if len(params) != len(self.position_to_param):
                err = f"Number of fit parameters ({len(params)}) does not match\n"
                err += f"number of unfixed parameters ({len(self.position_to_param)})\n"
                raise ValueError(err)

            for i in range(len(params)):
                self._mw_kwargs[self.position_to_param[i]] = params[i]

        try:
            return self._model_to_fit(**self._mw_kwargs)
        except Exception as e:
            err = "\n\nThe wrapped model threw an error (see trace).\n\n"
            raise type(e)(err) from e

    def load_fit_result(self,fitter):
        """
        Load the result of a fit into all fit parameters.
        """

        for i, p in enumerate(self.position_to_param):
            self.fit_parameters[p].load_fit_result(fitter,i)

    @property
    def model(self):
        """
        The observable.
        """

        # Update mapping between parameters and model arguments in case
        # user has fixed value
        self._update_parameter_map()

        # This model, once returned, does not have to re-run update_parameter_map
        # and should thus be faster when run again and again in regression
        return self._mw_observable

    @property
    def guesses(self):
        """
        Return an array of the guesses for the model (only including the unfixed
        parameters).
        """

        # Update mapping between parameters and model arguments in case
        # user has fixed value
        self._update_parameter_map()

        guesses = []
        for p in self.position_to_param:
            guesses.append(self.fit_parameters[p].guess)

        return np.array(guesses)

    @property
    def bounds(self):
        """
        Return an array of the bounds for the model (only including the unfixed
        parameters).
        """

        # Update mapping between parameters and model arguments in case
        # user has fixed value
        self._update_parameter_map()

        bounds = [[],[]]
        for p in self.position_to_param:
            bounds[0].append(self.fit_parameters[p].bounds[0])
            bounds[1].append(self.fit_parameters[p].bounds[1])

        return np.array(bounds)

    @property
    def priors(self):
        """
        Return an array of the priors for the model (only including the unfixed
        parameters).
        """

        # Update mapping between parameters and model arguments in case
        # user has fixed value
        self._update_parameter_map()

        priors = [[],[]]
        for p in self.position_to_param:
            priors[0].append(self.fit_parameters[p].prior[0])
            priors[1].append(self.fit_parameters[p].prior[1])

        return np.array(priors)

    @property
    def names(self):
        """
        Return an array of the names of the parameters (only including the unfixed
        parameters).
        """

        # Update mapping between parameters and model arguments in case
        # user has fixed value
        self._update_parameter_map()

        names = []
        for p in self.position_to_param:
            names.append(self.fit_parameters[p].name)

        return names[:]

    @property
    def fit_parameters(self):
        """
        A dictionary of FitParameter instances.
        """

        return self._mw_fit_parameters

    @property
    def other_arguments(self):
        """
        A dictionary with every model argument that is not a fit parameter.
        """

        return self._mw_other_arguments

    @property
    def position_to_param(self):
        """
        List mapping the position of each parameters in the output arrays to
        their original model argument names.
        """

        return self._position_to_param
