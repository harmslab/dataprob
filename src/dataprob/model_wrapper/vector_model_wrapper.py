
"""
Class for wrapping functions that use an array as their first argument for use
in likelihood calculations. 
"""

from dataprob.model_wrapper.model_wrapper import ModelWrapper

from dataprob.model_wrapper._function_processing import analyze_vector_input_fcn
from dataprob.model_wrapper._function_processing import param_sanity_check
from dataprob.model_wrapper._dataframe_processing import validate_dataframe


from dataprob.check import check_float

import numpy as np
import pandas as pd

class VectorModelWrapper(ModelWrapper):
    """
    Wrap a function that has an array as its first argument for use in 
    likelihood calculations.
    """

    def _load_model(self,
                    model_to_fit,
                    fittable_params):
        """
        Load a model into the wrapper, putting all fittable parameters into the
        param_df dataframe. Non-fittable arguments are set as attributes. 

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
        fit_params = []
        guesses = []
        for p in fittable_params:

            # If a dictionary, grab the guess checking for float
            if issubclass(type(fittable_params),dict):
                guess = check_float(value=fittable_params[p],
                                    variable_name=f"fittable_params['{p}']")
            
            # If a list, set to default_guess
            else:
                guess = self._default_guess
        
            # Record fit parameter
            fit_params.append(p)
            guesses.append(guess)
        
        self._fit_params_in_order = fit_params[:]
        param_df = pd.DataFrame({"name":fit_params,
                                 "guess":guesses})
        self._param_df = validate_dataframe(param_df,
                                            param_in_order=self._fit_params_in_order,
                                            default_guess=self._default_guess)

        # Set other argument values
        for p in other_args:
            self._other_arguments[p] = other_args[p]

        # Make sure these do not conflict with attributes already in the class
        reserved_params = dir(self.__class__)
        fittable_params = param_sanity_check(fittable_params=fittable_params,
                                             reserved_params=reserved_params)
        _ = param_sanity_check(fittable_params=other_args,
                               reserved_params=reserved_params)

        # Finalize -- read to run the model
        self.finalize_params()


    def finalize_params(self):
        """
        Validate current state of param_df and build map between parameters
        and the model arguments. This will be called by a Fitter instance 
        before doing a fit. 
        """
    
        # Make sure the parameter dataframe is sane. It could have problems 
        # because we let the user edit it directly.
        self._param_df = validate_dataframe(param_df=self._param_df,
                                            param_in_order=self._fit_params_in_order,
                                            default_guess=self._default_guess)
        
        # Get currently un-fixed parameters
        self._unfixed_mask = np.logical_not(self._param_df.loc[:,"fixed"])
        self._current_param_index = self._param_df.index[self._unfixed_mask]

    def _mw_observable(self,params=None):
        """
        Actual function called by the fitter.
        """

        compiled_params = np.array(self._param_df["guess"])

        if params is None:
            params = compiled_params

        if len(params) == len(compiled_params):
            compiled_params = params
        elif len(params) == np.sum(self._unfixed_mask):
            compiled_params[self._unfixed_mask] = params
        else:
            err = f"params length ({len(params)}) must either correspond to\n"
            err += f"the total number of parameters ({len(self._param_df)})\n"
            err += f"or the number of unfixed parameters ({np.sum(self._unfixed_mask)}).\n"
            raise ValueError(err)

        try:
            return self._model_to_fit(compiled_params,
                                      **self._other_arguments)
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
        self.finalize_params()

        # This model, once returned, does not have to re-run update_parameter_map
        # and should thus be faster when run again and again in regression
        return self._mw_observable