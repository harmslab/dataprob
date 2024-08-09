"""
Class for wrapping functions for use in likelihood calculations. 
"""

from dataprob.fit_param import FitParameter
from dataprob.check import check_array

from dataprob.model_wrapper._function_processing import analyze_fcn_sig
from dataprob.model_wrapper._function_processing import reconcile_fittable
from dataprob.model_wrapper._function_processing import param_sanity_check

from dataprob.model_wrapper.read_spreadsheet import load_param_spreadsheet

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

        # Define these here so __setattr__ and __getattr__ end up looking at
        # instance-level attributes rather than class-level attributes.
        self._mw_fit_parameters = {}
        self._mw_other_arguments = {}

        self._mw_load_model(model_to_fit,fittable_params)

    def _mw_load_model(self,model_to_fit,fittable_params):
        """
        Load a model into the wrapper, making the arguments into attributes.
        Fittable arguments are made into FitParameter instances.  Non-fittable
        arguments are set as generic attributes.

        Parameters
        ----------
        model_to_fit : callable
            a function or method to fit.
        fittable_params : list-like or None
            list of parameters to fit 
        """

        # Make sure input model is callable
        if not hasattr(model_to_fit,"__call__"):
            err = f"'{model_to_fit}' should be callable\n"
            raise ValueError(err)

        self._model_to_fit = model_to_fit

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

            # if there are kwargs, p will be in fittable_params but not in
            # can_be_fit.
            if p in can_be_fit:
                guess = can_be_fit[p]        
            else:
                guess = None

            self._mw_fit_parameters[p] = FitParameter(guess=guess)
        
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

        # Otherwise, just set it like normal
        else:
            super().__setattr__(key, value)

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

            # Look in dict for something set manually in instance
            if key in self.__dict__:
               return self.__dict__[key]

            # if not there, fall back to base __getattribute__
            return super().__getattribute__(key)

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
        Actual function called by the fitter. params are either None (saying 
        grab parameter values from fit_paramater.value) or an array of 
        float parameter values. 
        """

        # If parameters are not passed, stick in the current parameter
        # values
        if params is None:
            for p in self._position_to_param:
                self._mw_kwargs[p] = self.fit_parameters[p].value
        else:
            if len(params) != len(self._position_to_param):
                err = f"Number of fit parameters ({len(params)}) does not match\n"
                err += f"number of unfixed parameters ({len(self._position_to_param)})\n"
                raise ValueError(err)

            for i in range(len(params)):
                self._mw_kwargs[self._position_to_param[i]] = params[i]

        try:
            return self._model_to_fit(**self._mw_kwargs)
        except Exception as e:
            err = "\n\nThe wrapped model threw an error (see trace).\n\n"
            raise RuntimeError(err) from e

    def load_fit_result(self,fitter):
        """
        Load the result of a fit into all fit parameters. Parameters much match
        exactly between the two fits. 

        Parameters
        ----------
        fitter : dataprob.Fitter
            fitter instance that has had .fit() run previously. 
        """

        if not np.array_equal(fitter.names,self.names):
            err = "mismatch in the parameter names between the current model\n"
            err += "and the fit\n"
            raise ValueError(err)

        for i, p in enumerate(self._position_to_param):
            self.fit_parameters[p].load_fit_result(fitter,i)

    def load_param_dict(self,params_to_load):
        """
        Load parameter guesses, fixed-ness, bounds, and priors from a
        dictionary. 
        
        Parameters
        ----------
        params_to_load : dict
            Dictionary keys should be the names of parameters loaded into the
            model_wrapper. Values are themselves dictionaries keying attributes
            to their appropriate value. For example, the following argument:
                `param_to_load['K'] = {'fixed':True,'guess':5}`
            would fix parameter 'K' and set its guess to 5. Not all parameters
            and attributes need to be in the dictionary. Parameters not seen in 
            the model will cause an error. 
        
        Note
        ----
        Allowed attributes: 

        |----------+--------------------------------------------------------------------------|
        | key      | value                                                                    |
        |----------+--------------------------------------------------------------------------|
        | 'guess'  | single float value (must be within bounds, if specified)                 |
        | 'fixed'  | True of False                                                            | 
        | 'bounds' | (lower,upper) as floats (-np.inf,np.inf) allowed                         | 
        | 'prior'  | (mean,stdev) as floats (np.nan,np.nan) allowed, meaning uniform prior    |
        |----------+--------------------------------------------------------------------------| 

        """

        # make sure its a dictionary
        if not issubclass(type(params_to_load),dict):
            err = "params_to_load should be a dictionary keying parameter names\n"
            err += "to dictionaries of attribute values.\n"
            raise ValueError(err)

        # Set fit parameter attributes from the spreadsheet values
        for p in params_to_load:
            for field in params_to_load[p]:

                if p not in self.fit_parameters:
                    err = f"parameter '{p}' is not in this model\n"
                    raise ValueError(err)
                
                setattr(self.fit_parameters[p],field,params_to_load[p][field])
        
        # Update parameters with new information. 
        self._update_parameter_map()

    def load_param_spreadsheet(self,spreadsheet):
        """
        Load parameter guesses, fixed-ness, bounds, and priors from a
        spreadsheet. 

        Parameters
        ----------
        spreadsheet : str or pandas.DataFrame
            spreadsheet to read data from. If a string, the program will treat 
            it as a filename and attempt to read (xslx, csv, tsv, and txt) will
            be recognized. If a dataframe, values will be read directly from the
            dataframe

        Notes
        -----

        Allowable columns:

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

        + The 'param' column is required. All parameters in the spreadsheet must
          match parameters in the model; however, not all parameters in the
          model must be in the spreadsheet. Parameters not in the spreadsheet 
          retain their current features in the class. 

        + Parameter features specified in the spreadsheet will overwrite
          features already in the class. Features not set in the spreadsheet
          will not affect features in the class. (For example, you can safely 
          specify a spreadsheet with a 'guess' column for each parameter without
          altering priors set previously). 

        + If a 'guess' column is in the spreadsheet, all values must be finite
          and non-nan floats. 

        + If a 'fixed' column is in the spreadsheet, all values must be TRUE or
          FALSE.

        + Bounds are specified using the 'lower_bound' and 'upper_bound' columns.
          If only one is specified, the other is set to infinity. (For example,
          if there is an 'upper_bound' column, the lower bound is set to 
          -np.inf). Nan entries are interpreted as infinities. NOTE: you cannot
          set a lower bound in the spreadsheet while preserving upper bounds 
          already in the class (and vice versa). If you wish to set non-infinite
          bounds with a spreadsheet, you must specify both upper and lower 
          bounds.

        + Gaussian priors are specified using the 'prior_mean' and 'prior_std' 
          columns, declaring the prior mean and standard deviation. If either
          'prior_mean' or 'prior_std' is set to a non-nan value, both must be 
          set. If both are set to nan, the priors are set to uniform for that
          parameter. 
        """

        # Load spreadsheet into a dictionary
        params_to_load = load_param_spreadsheet(spreadsheet=spreadsheet)

        # Load via load_param_dict
        self.load_param_dict(params_to_load=params_to_load)

    @property
    def model(self):
        """
        Model observable. This function takes a numpy array the number of 
        unfixed parameters long. 

        Parameters
        ----------
        params : numpy.ndarray, optional
            float numpy array the length of the number of unfixed parameters.
            If this is not specified, the model is run using the values of each
            parameter (that is, the values seen in the "values" attribute). 
        """

        # Update mapping between parameters and model arguments in case
        # user has fixed value
        self._update_parameter_map()

        # This model, once returned, does not have to re-run update_parameter_map
        # and should thus be faster when run again and again in regression
        return self._mw_observable

    @property
    def names(self):
        """
        Names of unfixed parameters in the order they appear in the parameters
        array passed to model(param). 
        """

        # Update mapping between parameters and model arguments in case
        # user has fixed value. (NOTE: this update happens in the public
        # property. Most of the time, self._position_to_param should be accessed
        # by internal methods to avoid triggering this update.)
        self._update_parameter_map()

        return self._position_to_param
        

    @property
    def guesses(self):
        """
        Array of model guesses (only including unfixed parameters).
        """

        # Update mapping between parameters and model arguments in case
        # user has fixed value
        self._update_parameter_map()

        guesses = []
        for p in self._position_to_param:
            guesses.append(self.fit_parameters[p].guess)

        return np.array(guesses,dtype=float)
    
    @guesses.setter
    def guesses(self,guesses):

        # Update mapping between parameters and model arguments in case
        # user has fixed value
        self._update_parameter_map()

        n = len(self._position_to_param)

        guesses = check_array(value=guesses,
                              variable_name="guesses",
                              expected_shape=(n,),
                              expected_shape_names="(num_unfixed_param,)")
        for i, p in enumerate(self._position_to_param):
            self._mw_fit_parameters[p].guess = guesses[i]


    @property
    def bounds(self):
        """
        Array of parameter bounds (only including unfixed parameters).
        """

        # Update mapping between parameters and model arguments in case
        # user has fixed value
        self._update_parameter_map()

        bounds = [[],[]]
        for p in self._position_to_param:
            bounds[0].append(self.fit_parameters[p].bounds[0])
            bounds[1].append(self.fit_parameters[p].bounds[1])

        return np.array(bounds,dtype=float)
    
    @bounds.setter
    def bounds(self,bounds):

        # Update mapping between parameters and model arguments in case
        # user has fixed value
        self._update_parameter_map()

        n = len(self._position_to_param)

        bounds = check_array(value=bounds,
                             variable_name="bounds",
                             expected_shape=(2,n),
                             expected_shape_names="(2,num_unfixed_param)")
        for i, p in enumerate(self._position_to_param):
            self._mw_fit_parameters[p].bounds = bounds[:,i]


    @property
    def priors(self):
        """
        Array of the priors (only including unfixed parameters).
        """

        # Update mapping between parameters and model arguments in case
        # user has fixed value
        self._update_parameter_map()

        priors = [[],[]]
        for p in self._position_to_param:
            priors[0].append(self.fit_parameters[p].prior[0])
            priors[1].append(self.fit_parameters[p].prior[1])

        return np.array(priors,dtype=float)
    
    @priors.setter
    def priors(self,priors):

        # Update mapping between parameters and model arguments in case
        # user has fixed value
        self._update_parameter_map()

        n = len(self._position_to_param)

        priors = check_array(value=priors,
                             variable_name="priors",
                             expected_shape=(2,n),
                             expected_shape_names="(2,num_unfixed_param)")
        for i, p in enumerate(self._position_to_param):
            self._mw_fit_parameters[p].prior = priors[:,i]

    
    @property
    def fixed_mask(self):
        """
        Boolean array as long as the total number of parameters. True entries
        are fixed parameters; False entries are floating.
        """
        
        # Update mapping between parameters and model arguments in case
        # user has fixed value
        self._update_parameter_map()

        fixed_mask = []
        for p in self._mw_fit_parameters:
            fixed_mask.append(self._mw_fit_parameters[p].fixed)

        return np.array(fixed_mask,dtype=bool)
    
    @fixed_mask.setter
    def fixed_mask(self,fixed_mask):

        if not hasattr(fixed_mask,'__iter__'):
            err = "fixed_mask should be an bool array the same length as the\n"
            err += "total number of parameters\n"
            raise ValueError(err)
        
        if len(fixed_mask) != len(self._mw_fit_parameters):
            err = "fixed_mask should be an bool array the same length as the\n"
            err += "total number of parameters\n"
            raise ValueError(err)

        for i, p in enumerate(self._mw_fit_parameters):
            self._mw_fit_parameters[p].fixed = fixed_mask[i]

        # Update mapping between parameters and model arguments since we just
        # set fixedness
        self._update_parameter_map()

    @property
    def dataframe(self):
        """
        Parameters as a dataframe. Parameters can also be set using this
        property.

        ```
        # mw is a ModelWrapper instance
        df = mw.dataframe
        df.loc[0,"guess"] = 5
        mw.dataframe = df
        ```
        """

        # Update parameter mapping and model arguments to our dataframe is in
        # sync with the model 
        self._update_parameter_map()

        out = {"param":[],
               "guess":[],
               "fixed":[],
               "lower_bound":[],
               "upper_bound":[],
               "prior_mean":[],
               "prior_std":[]}
        
        for p in self._mw_fit_parameters:

            out["param"].append(p)
            
            fp = self._mw_fit_parameters[p]

            out["guess"].append(fp.guess)
            out["fixed"].append(fp.fixed)
            out["lower_bound"].append(fp.bounds[0])
            out["upper_bound"].append(fp.bounds[1])
            out["prior_mean"].append(fp.prior[0])
            out["prior_std"].append(fp.prior[1])

        return pd.DataFrame(out)

    
    @dataframe.setter
    def dataframe(self,dataframe):
        
        # Setter is a convenience wrapper for load_param_spreadsheet.
        self.load_param_spreadsheet(dataframe)


    @property
    def fit_parameters(self):
        """
        A dictionary of FitParameter instances, including both fixed and 
        unfixed parameters. 
        """

        return self._mw_fit_parameters

    @property
    def other_arguments(self):
        """
        A dictionary with every model argument that is not a fit parameter.
        """

        return self._mw_other_arguments


    def __repr__(self):
        """
        Useful summary of current model wrapper state.
        """
        
        self._update_parameter_map()

        out = ["ModelWrapper\n------------\n"]

        # model name
        out.append(f"wrapped_model: {self._model_to_fit.__name__}\n")

        # Non fittable arguments
        out.append(f"  non-fittable arguments:\n")
        for p in self._mw_other_arguments:
            out.append(f"    {p}:")

            # See if there are multiple lines on this repr...
            variable_lines = repr(self._mw_other_arguments[p]).split("\n")
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
        for dataframe_line in repr(self.dataframe).split("\n"):
            out.append(f"    {dataframe_line}")
        out.append("\n")

        return "\n".join(out)
