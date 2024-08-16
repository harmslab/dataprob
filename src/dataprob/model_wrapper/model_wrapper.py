"""
Class for wrapping functions for use in likelihood calculations. 
"""

from dataprob.model_wrapper._function_processing import analyze_fcn_sig
from dataprob.model_wrapper._function_processing import reconcile_fittable
from dataprob.model_wrapper._function_processing import param_sanity_check

from dataprob.model_wrapper._dataframe_processing import read_spreadsheet
from dataprob.model_wrapper._dataframe_processing import validate_dataframe
from dataprob.model_wrapper._dataframe_processing import param_into_existing

from dataprob.check import check_float

import numpy as np
import pandas as pd

class ModelWrapper:
    """
    Wrap a function for use in likelihood calculations.

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
    _param_df = pd.DataFrame({"name":[]})
    _other_arguments = {}

    def __init__(self,
                 model_to_fit,
                 fittable_params=None,
                 non_fit_kwargs=None,
                 default_guess=0.0):
        """
        Parameters
        ----------
        model_to_fit : callable
            a function or method to fit.
        fittable_params : list-like, optional
            list of arguments to fit.
        non_fit_kwargs : dict
            non_fit_kwargs are keyword arguments for model_to_fit that should
            be fit but need to be specified to non-default values. 
        default_guess : float, default=0
            assign parameters with no default value this value
        """

        # Make sure input model is callable
        if not hasattr(model_to_fit,"__call__"):
            err = f"'{model_to_fit}' should be callable\n"
            raise ValueError(err)

        # Check fittable_params
        if fittable_params is not None:
            if not hasattr(fittable_params,"__iter__") or issubclass(type(fittable_params),str):
                err = "fittable_parameters should be a list of parameter names\n"
                raise ValueError(err)
        
        # check non_fit_kwargs
        if non_fit_kwargs is not None:
            if not issubclass(type(non_fit_kwargs),dict):
                err = "non_fit_kwargs must be a dictionary of keyword arguments\n"
                err += "to be passed to model_to_fit when the function is run.\n"
                raise ValueError(err)

        self._default_guess = check_float(value=default_guess,
                                          variable_name="default_guess")

        # Re-define these here so __setattr__ and __getattr__ end up looking at
        # instance-level (__dict__) attributes rather than class-level
        # attributes.
        self._param_df = pd.DataFrame({"name":[]})
        self._other_arguments = {}

        self._load_model(model_to_fit=model_to_fit,
                         fittable_params=fittable_params,
                         non_fit_kwargs=non_fit_kwargs)
        

    def _load_model(self,model_to_fit,fittable_params,non_fit_kwargs):
        """
        Load a model into the wrapper. Fittable arguments are put into param_df.
        Non-fittable arguments are placed in the other_arguments dictionary.

        Parameters
        ----------
        model_to_fit : callable
            a function or method to fit.
        fittable_params : list-like or None
            list of parameters to fit 
        non_fit_kwargs : dict
            non_fit_kwargs are keyword arguments for model_to_fit that should
            be fit but need to be specified to non-default values. 
        """

        self._model_to_fit = model_to_fit

        if non_fit_kwargs is None:
            non_fit_kwargs = {}

        # Parse function arguments
        all_args, can_be_fit, cannot_be_fit, has_kwargs = \
            analyze_fcn_sig(fcn=self._model_to_fit)

        # Decide which parameters are fittable and which are not
        fittable_params, not_fittable_parameters = \
            reconcile_fittable(fittable_params=fittable_params,
                               non_fit_kwargs=non_fit_kwargs,
                               all_args=all_args,
                               can_be_fit=can_be_fit,
                               cannot_be_fit=cannot_be_fit,
                               has_kwargs=has_kwargs)

        # Make sure input arguments are sane and compatible with the ModelWrapper
        # class namespace
        reserved_params = dir(self.__class__)
        fittable_params = param_sanity_check(param_to_check=fittable_params,
                                             reserved_params=reserved_params)
        non_fit_kwargs = param_sanity_check(param_to_check=non_fit_kwargs,
                                            reserved_params=reserved_params)

        # Go through fittable params.
        fit_params = []
        guesses = []
        for p in fittable_params:

            fit_params.append(p)

            # if **kwargs is defined, p could be in fittable_params but not in
            # can_be_fit.
            if p in can_be_fit:
                guesses.append(can_be_fit[p])
            else:
                guesses.append(None)

        # Remove any 'None' guesses and replace with default
        final_guesses = []
        for g in guesses:
            if g is None:
                final_guesses.append(self._default_guess)
            else:
                final_guesses.append(g)

        # Fix the order of the fit parameters
        self._fit_params_in_order = fit_params[:]
        
        # Build a param_df dataframe
        param_df = pd.DataFrame({"name":fit_params,
                                 "guess":final_guesses})
        self._param_df = validate_dataframe(param_df,
                                            param_in_order=self._fit_params_in_order,
                                            default_guess=self._default_guess)

        # Go through non-fittable parameters and record their keyword arguments
        # in _other_arguments. Look in 'can_be_fit', then 'cannot_be_fit'. If
        # no default argument from either of those, set to 'None'. Finally, 
        # look in 'non_fit_kwargs.' If defined here, it will take precedence 
        # over the default values. 
        for p in not_fittable_parameters:
            
            if p in can_be_fit:
                non_fit_param_value = can_be_fit[p]
            elif p in cannot_be_fit:
                non_fit_param_value = cannot_be_fit[p]
            else:
                non_fit_param_value = None

            if p in non_fit_kwargs:
                non_fit_param_value = non_fit_kwargs[p]
                
            self._other_arguments[p] = non_fit_param_value
          
        # Finalize -- read to run the model
        self.finalize_params()


    def __setattr__(self, key, value):
        """
        Hijack __setattr__ so setting the value for fit parameters
        updates the fit guess.
        """

        # We're setting the guess of the fit parameter
        if key in self._param_df.name:

            tmp_param_df = self._param_df.copy()
            tmp_param_df.loc[key,"guess"] = check_float(value=value,
                                                        variable_name="guess")
            self._param_df = validate_dataframe(tmp_param_df,
                                                param_in_order=self._fit_params_in_order,
                                                default_guess=self._default_guess)

        # We're setting another argument
        elif key in self._other_arguments.keys():
            self._other_arguments[key] = value

        # Otherwise, just set it like normal
        else:
            super().__setattr__(key, value)

    def __getattr__(self,key):
        """
        Define __getattr__ to get fit parameters and other arguments
        appropriately.
        """

        # We're getting a fit parameter
        if key in self._param_df.name:
            return self._param_df.loc[key,"guess"]

        # We're getting another argument
        if key in self._other_arguments.keys():
            return self._other_arguments[key]

        # Otherwise, get like normal
        else:

            # Look in dict for something set manually in instance
            if key in self.__dict__:
               return self.__dict__[key]

            # if not there, fall back to base __getattribute__
            return super().__getattribute__(key)

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
        self._unfixed_param_names = np.array(self._param_df.loc[self._unfixed_mask,"name"])

        # Build a dictionary of keyword arguments to pass to the model when
        # called. 
        self._mw_kwargs = {}
        for p in self._fit_params_in_order:
            self._mw_kwargs[p] = self._param_df.loc[p,"guess"]
        self._mw_kwargs.update(self._other_arguments)

    def _mw_observable(self,params=None):
        """
        Actual function called by the fitter. 
        """

        # If parameters are not passed, get current parameter values
        if params is None:
            params = np.array(self._param_df.loc[self._unfixed_mask,
                                                 "guess"],dtype=float)
        else:
            params = np.array(params,dtype=float)

        # Sanity check
        if len(params) != np.sum(self._unfixed_mask):
            err = f"Number of fit parameters ({len(params)}) does not match\n"
            err += f"number of unfixed parameters ({np.sum(self._unfixed_mask)})\n"
            raise ValueError(err)

        # Update kwargs
        for i in range(len(params)):
            self._mw_kwargs[self._unfixed_param_names[i]] = params[i]

        try:
            return self._model_to_fit(**self._mw_kwargs)
        except Exception as e:
            err = "\n\nThe wrapped model threw an error (see trace).\n\n"
            raise RuntimeError(err) from e
        
    def update_params(self,param_input):
        """
        Update the parameter features. 

        Parameters
        ----------
        param_input : pandas.DataFrame or dict or str
            param_input should hold parameter features. If param_input is a
            string, it will be treated as a filename and read by pandas (xslx,
            csv, tsv, and txt are recognized). If param_input is a dataframe,
            values will be read directly from the dataframe. If param_input is a
            dict, it will be treated as a nested dictionary keying parameter
            names to columns to values (param_input[parameter][column] -> value).
            (Example: ``param_input={"K":{"guess":1.0}}`` would set the guess
            for parameter "K" to 1.0).

        Notes
        -----
        See the param_df docstring for details on parameter inputs. 

        Parameter features specified in param_input will overwrite features in
        param_df. Features *not* set in param_input will *not* alter existing
        features in param_df. For example, you can safely specify a spreadsheet
        with a 'guess' column without altering priors already set in param_df. 
        Or, you could send in a dictionary setting the lower_bound for a single
        parameter without altering any other parameters. 
        """

        # If a string, read as a spreadsheet
        if issubclass(type(param_input),str):
            param_input = read_spreadsheet(param_input)

        # Load the parameter input into the dataframe
        param_df = param_into_existing(param_input=param_input,
                                       param_df=self._param_df)
        
        # Validate the final dataframe and store it
        self._param_df = validate_dataframe(param_df=param_df,
                                            param_in_order=self._fit_params_in_order,
                                            default_guess=self._default_guess)

    @property
    def model(self):
        """
        Model observable. This function takes a numpy array the number of 
        unfixed parameters long. 

        Parameters
        ----------
        params : numpy.ndarray, optional
            float numpy array the length of the number of unfixed parameters.
            If this is not specified, the model is run using the parameter 
            guess values. 
        """

        self.finalize_params()

        # This model, once returned, does not have to re-run finalize and should
        # thus be faster when run again and again in regression
        return self._mw_observable

    @property
    def param_df(self):
        """
        Dataframe holding the fittable parameters in the model. This can be set 
        by ``mw.param_df = some_new_df``. It can also be edited in place 
        (e.g. ``mw.param_df.loc["K1","guess"] = 5``).
        
        The 'name' column is set when the dataframe is initialized. This defines
        the names of the parameters, which cannot be changed later. The 'name'
        column is used as the index for the dataframe. 

        This dataframe will minimally have the following columns. Other
        columns may be present if set by the user, but will be ignored. 

        +---------------+-----------------------------------------------------+
        | key           | value                                               |
        +===============+=====================================================+
        | 'name'        | string name of the parameter. should not be changed |
        |               | by the user.                                        |
        +---------------+-----------------------------------------------------+
        | 'guess'       | guess as single float value (must be non-nan and    |
        |               | within bounds if specified)                         |
        +---------------+-----------------------------------------------------+
        | 'fixed'       | whether or not parameter can vary. True of False    |
        +---------------+-----------------------------------------------------+
        | 'lower_bound' | single float value; -np.inf allowed; None, nan or   |
        |               | pd.NA interpreted as -np.inf.                       |
        +---------------+-----------------------------------------------------+
        | 'upper_bound' | single float value; -np.inf allowed; None, nan or   |
        |               | pd.NA interpreted as np.inf.                        |
        +---------------+-----------------------------------------------------+
        | 'prior_mean'  | single float value; np.nan allowed (see below)      |
        +---------------+-----------------------------------------------------+
        | 'prior_std'   | single float value; np.nan allowed (see below)      |
        +---------------+-----------------------------------------------------+

        Gaussian priors are specified using the 'prior_mean' and 'prior_std' 
        fields, declaring the prior mean and standard deviation. If both are
        set to nan for a parameter, the prior is set to uniform between the
        parameter bounds. If either 'prior_mean' or 'prior_std' is set to a
        non-nan value, both must be non-nan to define the prior. When set, 
        'prior_std' must be greater than zero. Neither can be np.inf. 
        """
        
        return self._param_df
    
    @param_df.setter
    def param_df(self,param_df):

        # Validate the parameter dataframe before setting it
        self._param_df = validate_dataframe(param_df,
                                            param_in_order=self._fit_params_in_order)
        
    @property
    def other_arguments(self):
        """
        A dictionary with every model argument that is not a fit parameter.
        """

        return self._other_arguments
    
    @property
    def unfixed_mask(self):
        """
        Mask for param_df that returns only floating (unfixed) parameters.
        """

        return self._unfixed_mask
        
    def __repr__(self):
        """
        Useful summary of current model wrapper state.
        """
        
        self.finalize_params()

        out = []
        out = ["ModelWrapper\n------------\n"]

        # model name
        out.append(f"  wrapped_model: {self._model_to_fit.__name__}\n")

        # Non fittable arguments
        out.append(f"  non-fittable arguments:\n")
        for p in self._other_arguments:
            out.append(f"    {p}:")

            # See if there are multiple lines on this repr...
            variable_lines = repr(self._other_arguments[p]).split("\n")
            if len(variable_lines) > 6:
                to_add = variable_lines[:3]
                to_add.append("...")
                to_add.extend(variable_lines[-3:])
            else:
                to_add = variable_lines

            for line in to_add:
                out.append(f"      {line}")

        out.append("\n")

        # Fittable arguments
        out.append(f"  fittable parameters:\n")
        for dataframe_line in repr(self.param_df).split("\n"):
            out.append(f"    {dataframe_line}")
        out.append("\n")

        return "\n".join(out)
