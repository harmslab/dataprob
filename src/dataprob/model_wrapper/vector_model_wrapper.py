"""
Class for wrapping functions that use an array as their first argument for use
in likelihood calculations. 
"""

from dataprob.model_wrapper.model_wrapper import ModelWrapper

from dataprob.model_wrapper._function_processing import analyze_vector_input_fcn
from dataprob.model_wrapper._dataframe_processing import validate_dataframe


from dataprob.util.check import check_float

import numpy as np
import pandas as pd

class VectorModelWrapper(ModelWrapper):
    """
    Wrap a function that has an array as its first argument for use in 
    likelihood calculations.
    """

    def _load_model(self,
                    model_to_fit,
                    fit_parameters,
                    non_fit_kwargs):
        """
        Load a model into the wrapper, putting all fittable parameters into the
        param_df dataframe. Non-fittable arguments are set as attributes. 

        Parameters
        ----------
        model_to_fit : callable
            a function or method to fit.
        fit_parameters : list or dict
            dictionary of fit parameters with guesses
        non_fit_kwargs : dict
            non_fit_kwargs are keyword arguments for model_to_fit that should
            be fit but need to be specified to non-default values. 
        """

        self._model_to_fit = model_to_fit

        # Parse function
        param_arg, other_args, has_kwargs = analyze_vector_input_fcn(self._model_to_fit)

        # Make sure it has at least one argument
        if param_arg is None:
            err = f"model '{self._model_to_fit}' should take at least one argument\n"
            raise ValueError(err)
        
        # Make sure fittable params has at least one param
        try:
            num_param = len(fit_parameters)
            if num_param < 1:
                raise ValueError
        except Exception as e:
            err = f"fit_parameters must be a list or dictionary with at least one\n"
            err += "fittable parameter\n"
            raise ValueError(err) from e

        # Make sure fittable param names do not conflict with argument param
        # names
        fit_set = set(fit_parameters)
        args_set = set(other_args)
        if len(fit_set.intersection(args_set)) > 0:
            err = "fit_parameters must not include other arguments to the function\n"
            raise ValueError(err)
        
        if param_arg in fit_parameters:
            err = f"the first vector arg '{param_arg}' cannot be in fit_parameters.\n"
            err += "fit_parameters should specify the names of every element\n"
            err += "*within* this vector\n"
            raise ValueError(err)

        # --------------------------------------------------------------------
        # Go through fittable params 

        fit_param_names = []
        guesses = []
        for p in fit_parameters:

            # If a dictionary, grab the guess checking for float
            if issubclass(type(fit_parameters),dict):
                guess = check_float(value=fit_parameters[p],
                                    variable_name=f"fit_parameters['{p}']")
            
            # If a list, set to default_guess
            else:
                guess = self._default_guess
        
            # Record fit parameter
            fit_param_names.append(p)
            guesses.append(guess)
        
        # Construct fit parameter dataframe
        self._fit_params_in_order = fit_param_names[:]
        param_df = pd.DataFrame({"name":fit_param_names,
                                 "guess":guesses})
        self._param_df = validate_dataframe(param_df,
                                            param_in_order=self._fit_params_in_order,
                                            default_guess=self._default_guess)

        # --------------------------------------------------------------------
        # Deal with non_fit_kwargs

        # Construct not_fit_parameters from non_fit_kwargs keys
        if non_fit_kwargs is not None:
            not_fit_parameters = list(non_fit_kwargs.keys())
        else:
            not_fit_parameters = []

        # Make sure param_arg is not in not_fit_parameters
        if param_arg in not_fit_parameters:
            err = f"the first argument {param_arg} cannot be in non_fit_kwargs\n"
            raise ValueError(err)

        # Go through non-fittable parameters. If they are not in other_args 
        # and the function does not have **kwargs, throw an error. 
        for p in not_fit_parameters:
            if p not in other_args and not has_kwargs:
                err = f"not_fittable parameter '{p}' is not in the function definition\n"
                raise ValueError(err)
            
            # If we get here, overwrite whatever was in other_args (default from
            # signature) with what hte user passed in. 
            other_args[p] = non_fit_kwargs[p]
                
        # Make sure that we don't have a situation where we have the same 
        # parameter name in both fittable and not_fittable
        fittable_set = set(fit_parameters)
        not_fittable_set = set(other_args)
        intersect = fittable_set.intersection(not_fittable_set)
        if len(intersect) != 0:
            err = "a parameter cannot be in both fit_parameters and non_fit_kwargs.\n"
            err += f"Bad parameters: {str(intersect)}\n"
            raise ValueError(err)
        
        # Store the non fit parameter values. 
        self._non_fit_kwargs_keys = []
        for p in other_args:
            self._non_fit_kwargs[p] = other_args[p]
            self._non_fit_kwargs_keys.append(p)
        
        # This set holds the expected set of kwargs keys. This allows us to 
        # make sure the user does not add or remove a key with the setter. 
        self._non_fit_kwargs_keys = set(self._non_fit_kwargs_keys)

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
        self._unfixed_mask = np.array(np.logical_not(self._param_df["fixed"]),dtype=bool)
        self._unfixed_param_names = np.array(self._param_df.loc[self._unfixed_mask,"name"]).copy()
        
        # Create all param vector
        self._all_param_vector = np.array(self._param_df["guess"],dtype=float).copy()
    
        # Make sure the user has not altered non_fit_kwargs keys
        self._validate_non_fit_kwargs()


    def model(self,params=None):
        """
        Model observable. This function takes a numpy array either the number of 
        unfixed parameters long OR the total number of parameters long. If 
        parameters are fixed, their values in a params array with all fit 
        parameters are *ignored* and the fixed parameter guesses are used 
        instead. 

        Parameters
        ----------
        params : numpy.ndarray, optional
            float numpy array with parameter values. If this is not specified,
            the model is run using the parameter guess values. 
        """
        
        # Update mapping between parameters and model arguments in case
        # user has fixed value or made a change that has not propagated properly
        self.finalize_params()

        compiled_params = np.array(self._param_df["guess"],dtype=float).copy()

        if params is None:
            params = compiled_params

        # make sure the params are a float array
        params = np.array(params,dtype=float)

        # Copy in only unfixed params from full vector sent in
        if len(params) == len(compiled_params):
            compiled_params[self._unfixed_mask] = params[self._unfixed_mask]

        # Copy in all params into unfixed positions
        elif len(params) == np.sum(self._unfixed_mask):
            compiled_params[self._unfixed_mask] = params
        else:
            err = f"params length ({len(params)}) must either correspond to\n"
            err += f"the total number of parameters ({len(self._param_df)})\n"
            err += f"or the number of unfixed parameters ({np.sum(self._unfixed_mask)}).\n"
            raise ValueError(err)

        try:
            return self._model_to_fit(compiled_params,
                                      **self._non_fit_kwargs)
        except Exception as e:
            err = "\n\nThe wrapped model threw an error (see trace).\n\n"
            raise RuntimeError(err) from e
    
    def fast_model(self,params):
        """
        Calculate model result with minimal error checking. 

        Parameters
        ----------
        params : numpy.ndarray
            vector of unfixed parameter values

        Returns
        -------
        out : numpy.ndarray
            result of model(params)
        """

        self._all_param_vector[self._unfixed_mask] = params
        return self._model_to_fit(self._all_param_vector,
                                  **self._non_fit_kwargs)
