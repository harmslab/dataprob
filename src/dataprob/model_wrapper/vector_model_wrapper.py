

from dataprob.model_wrapper.model_wrapper import ModelWrapper
from dataprob.fit_param import FitParameter
from dataprob.model_wrapper._function_processing import analyze_vector_input_fcn
from dataprob.model_wrapper._function_processing import param_sanity_check
from dataprob.check import check_float

import numpy as np

class VectorModelWrapper(ModelWrapper):

    def _mw_load_model(self,model_to_fit,fittable_params):
        """
        Load a model into the wrapper, making the arguments into attributes.
        Fittable arguments are made into FitParameter instances.  Non-fittable
        arguments are set as generic attributes.

        Parameters
        ----------
        model_to_fit : callable
            a function or method to fit.
        fittable_params : list or dict
            dictionary of fit parameters with guesses
        """

        # Make sure input model is callable
        if not hasattr(model_to_fit,"__call__"):
            err = f"'{model_to_fit}' should be callable\n"
            raise ValueError(err)

        self._model_to_fit = model_to_fit

        # Parse function
        param_arg, other_args = analyze_vector_input_fcn(self._model_to_fit)

        # Make sure it has at least one argument
        if param_arg is None:
            err = f"model '{self._model_to_fit}' should take at least one argument\n"
            raise ValueError(err)
        
        # Make sure fittable params has at least one param
        try:
            num_param = len(fittable_params)
            if num_param < 1:
                raise ValueError
        except Exception as e:
                err = f"fittable_params must be a list or dictionary with at least one\n"
                err += "fittable parameter\n"
                raise ValueError(err) from e

        # Make sure fittable param names do not conflict with argument param
        # names
        fit_set = set(fittable_params)
        args_set = set(other_args)
        if len(fit_set.intersection(args_set)) > 0:
            err = "fittable_params must not include other arguments to the function\n"
            raise ValueError(err)

        # Go through fittable params   
        for p in fittable_params:

            # If a dictionary, grab the guess checking for float
            if issubclass(type(fittable_params),dict):
                guess = check_float(value=fittable_params[p],
                                    variable_name=f"fittable_params['{p}']")
            
            # If a list, guess is 0.0
            else:
                guess = 0
        
            # Record fit parameter
            self._mw_fit_parameters[p] = FitParameter(name=p,guess=guess)
        
        # Set other argument values
        for p in other_args:
            self._mw_other_arguments[p] = other_args[p]

        # Make sure these do not conflict with attributes already in the class
        reserved_params = dir(self.__class__)
        fittable_params = param_sanity_check(fittable_params=fittable_params,
                                             reserved_params=reserved_params)
        _ = param_sanity_check(fittable_params=other_args,
                               reserved_params=reserved_params)

        self._update_parameter_map()


    def _update_parameter_map(self):
        """
        Update the map between the parameter vector that will be passed in to
        the fitter and the parameters in this wrapper. 
        """

        param_vector_length = len(self._mw_fit_parameters)

        num_unfixed = param_vector_length
        unfixed_param_mask = np.ones(param_vector_length,dtype=bool)

        current_parameter_values = []
        self._position_to_param = []
        for i, p in enumerate(self._mw_fit_parameters):

            if self._mw_fit_parameters[p].fixed:
                unfixed_param_mask[i] = False
                num_unfixed -= 1

            current_parameter_values.append(self._mw_fit_parameters[p].guess)
            self._position_to_param.append(p)

        self._param_vector_length = param_vector_length
        self._num_unfixed = num_unfixed
        self._unfixed_param_mask = unfixed_param_mask
        self._current_parameter_values = np.array(current_parameter_values,
                                                  dtype=float)


    def _mw_observable(self,params=None):
        """
        Actual function called by the fitter.
        """

        if params is None:
            params = self._current_parameter_values

        if len(params) == self._param_vector_length:
            compiled_params = params
        elif len(params) == self._num_unfixed:
            compiled_params = self._current_parameter_values.copy()
            compiled_params[self._unfixed_param_mask] = params
        else:
            err = f"params length ({len(params)}) must either correspond to\n"
            err += f"the total number of parameters ({self._param_vector_length})\n"
            err += f"or the number of unfixed parameters ({self._num_unfixed}).\n"
            raise ValueError(err)

        try:
            return self._model_to_fit(compiled_params,**self._mw_other_arguments)
        except Exception as e:
            err = "\n\nThe wrapped model threw an error (see trace).\n\n"
            raise RuntimeError(err) from e

    @property
    def model(self):
        """
        The observable.
        """

        # Update mapping between parameters and model arguments in case
        # user has fixed value or made a change that has not propagated properly
        self._update_parameter_map()

        # This model, once returned, does not have to re-run update_parameter_map
        # and should thus be faster when run again and again in regression
        return self._mw_observable