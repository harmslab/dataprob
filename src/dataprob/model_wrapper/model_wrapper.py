"""
Class for wrapping models for use in likelihood calculations. 
"""

from dataprob.fit_param import FitParameter
from dataprob.model_wrapper._function_processing import analyze_fcn_sig
from dataprob.model_wrapper._function_processing import reconcile_fittable
from dataprob.model_wrapper._function_processing import param_sanity_check

import numpy as np


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

        # Make sure input model is callable
        if not hasattr(model_to_fit,"__call__"):
            err = f"'{model_to_fit}' should be callable\n"
            raise ValueError(err)

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

        all_args, can_be_fit, cannot_be_fit, has_kwargs = \
            analyze_fcn_sig(fcn=self._model_to_fit)

        fittable_params, not_fittable_parameters = \
            reconcile_fittable(fittable_params=fittable_params,
                               all_args=all_args,
                               can_be_fit=can_be_fit,
                               cannot_be_fit=cannot_be_fit,
                               has_kwargs=has_kwargs)

        reserved_params = dir(self.__class__)
        fittable_params = param_sanity_check(fittable_params=fittable_params,
                                             reserved_params=reserved_params)
        not_fittable_parameters = param_sanity_check(fittable_params=not_fittable_parameters,
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
        the fitter and the parameters in this wrapper. 
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

