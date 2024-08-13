"""
Fitter base class allowing different classes of fits.
"""

import numpy as np
import pandas as pd
import corner


import re
import pickle
import os
import warnings

from dataprob.check import check_array
from dataprob.check import check_float
from dataprob.check import check_int

from dataprob.model_wrapper.model_wrapper import ModelWrapper
from dataprob.model_wrapper.vector_model_wrapper import VectorModelWrapper

def _pretty_zeropad_str(N):

    num_digits = len(f"{N}") + 1
    fmt_string = "s{:0" + f"{num_digits}" + "d}"
    return fmt_string

class Fitter:
    """
    Base class for fits/analyses using a dataprob function.
    """

    def __init__(self):
        """
        Init function for the class.
        """

        # This is set to True after a fit is run but is reset to False by
        # the setter functions like guesses, etc. that would influence the
        # fit results.
        self._fit_has_been_run = False
        self._fit_type = ""

    def _sanity_check(self,call_string,attributes_to_check):
        """
        Do a sanity check before doing model calculations.

        Parameters
        ----------
        call_string : str
            string to dump into output on error.
        attributes_to_check : list
            list of attributes to check
        """

        for a in attributes_to_check:
            try:
                if self.__dict__[f"_{a}"] is None:
                    raise KeyError
            except KeyError:
                err = f"'{a}' must be set before {call_string}\n"
                raise RuntimeError(err)

    def _process_model_args(self,model,guesses,names):
        """
        Process the arguments from .fit that define the model. Figure out 
        whether to simply store the model as passed in or to wrap the model 
        in a ModelWrapper instance. Validates consistency of model, guesses,
        and names. 
        """

        # Sanity check. If model is already defined, do not let the user send
        # in model or names arguments. Guesses are fine -- these will be 
        # processed later. 
        if self.model is not None:

            if model is not None:
                err = "A model should not be passed to fit() if the model has\n"
                err += "already been defined with the model setter.\n"
                raise ValueError(err)

            if names is not None:
                err = "Parameter names cannot be specified in fit() if the\n"
                err += "model has already been defined with the model setter.\n"
                raise ValueError(err)
            
            return
        
        # If self.model is not defined, make sure the user specifies a model. 
        if model is None:
            err = "model must be specified prior to fit\n"
            raise ValueError(err)
            
        # Easy case. ModelWrapper being passed in, so just assign it to 
        # self._model. Throw an error if the user sent in names because you 
        # cannot set ModelWrapper names after the object is initialized. 
        if issubclass(type(model),ModelWrapper):

            if names is not None:
                err = "Parameter names cannot be specified after creating a\n"
                err += "wrapper. Omit the 'names' argument when calling fit\n"
                err += "with this model.\n"
                raise ValueError(err)

            self.model = model
            return

        # Harder case. This is a naked function -- we'll have to build our
        # parameters based on the user input. 
        if guesses is None:
            err = "guesses must be specified when passing in an unwrapped\n"
            err += "function as a model.\n"
            raise ValueError(err)
        
        # Make sure the guesses are sane for figuring out the number of 
        # parameters. (That's all this is used for here; we set guesses
        # later).
        guesses = check_array(value=guesses,
                              variable_name="guesses",
                              expected_shape=(None,),
                              expected_shape_names="(num_params,)")

        # If no names are specified, make up parameter names p0, p1, etc.
        if names is None:
            names = ["p{}".format(i) for i in range(len(guesses))]

        if len(names) != len(guesses):
            err = "If both are specified, names and guesses should be the same\n"
            err += "length.\n"
            raise ValueError(err)

        # This wil create model, checking it for sanity
        mw = VectorModelWrapper(model_to_fit=model,
                                fittable_params=names)
        
        # Final assignment.
        self.model = mw


    def _process_fit_args(self,
                          guesses,
                          lower_bounds,
                          upper_bounds,
                          prior_means,
                          prior_stds,
                          fixed):
        """
        Process the arguments the user sends in setting the fit parameters. 
        Updates self.param_df with the new values, only setting if the values
        are consistent with each other (guesses between bounds, etc.)
        """
        
        # Create a copy of the ModelWrapper parameter dataframe. Populate this
        # with guesses, lower_bounds, upper_bounds, prior_means, prior_stds, 
        # and fixedness, then drop it back in fully populated with the inputs.
        # Doing it this way allows temporary inconsistent states while being set
        # (guesses outside bounds, for example), but makes sure the final
        # dataframe is consistent by passing it through the ModelWrapper.param_df
        # setter. 
        param_df = self._model.param_df.copy()

        num_params = len(param_df)

        if guesses is not None:
            guesses = check_array(value=guesses,
                                  variable_name="guesses",
                                  expected_shape=(num_params,),
                                  expected_shape_names=f"({num_params},)")
            param_df.loc[:,"guess"] = guesses
        
        if lower_bounds is not None:
            lower_bounds = check_array(value=lower_bounds,
                                       variable_name="lower_bounds",
                                       expected_shape=(num_params,),
                                       expected_shape_names=f"({num_params},)")
            param_df.loc[:,"lower_bound"] = lower_bounds

        if upper_bounds is not None:
            upper_bounds = check_array(value=upper_bounds,
                                       variable_name="upper_bounds",
                                       expected_shape=(num_params,),
                                       expected_shape_names=f"({num_params},)")
            param_df.loc[:,"upper_bound"] = upper_bounds

        if prior_means is not None:
            prior_means = check_array(value=prior_means,
                                      variable_name="prior_means",
                                      expected_shape=(num_params,),
                                      expected_shape_names=f"({num_params},)")
            param_df.loc[:,"prior_mean"] = prior_means

        if prior_stds is not None:
            prior_stds = check_array(value=prior_stds,
                                     variable_name="prior_stds",
                                     expected_shape=(num_params,),
                                     expected_shape_names=f"({num_params},)")
            param_df.loc[:,"prior_std"] = prior_stds

        if fixed is not None:
            fixed = check_array(value=fixed,
                                variable_name="fixed",
                                expected_shape=(num_params,),
                                expected_shape_names=f"({num_params},)")
            fixed = np.array(fixed,dtype=bool)
            param_df.loc[:,"fixed"] = fixed

        # Record guesses, etc. with the param_df setter. It will check for 
        # argument sanity. 
        self._model.param_df = param_df
        
    def _process_obs_args(self,
                          y_obs,
                          y_std):
        """
        Process the arguments the user sends in defining the observable (y_obs
        and y_std). Make sure they are consistent with each other.
        """

        # Record y_obs if specified. The setter does sanity checking. 
        if y_obs is not None:
            self.y_obs = y_obs
        
        # Make sure y_obs was specified
        if self.y_obs is None:
            err = "y_obs must be specified prior to doing a fit\n"
            raise ValueError(err)
        
        # Record y_std if specified. The setter does sanity checking. 
        if y_std is not None:
            self.y_std = y_std

        # Make fake y_std if not specified, warning. 
        if self.y_std is None:

            scalar = np.mean(np.abs(self._y_obs))*0.1
            self.y_std = scalar*np.ones(len(self.y_obs),dtype=float)

            w = "\ny_std must be sent in for proper residual/likelihood\n"
            w += f"calculation. We have arbitrarily set it to {scalar:.2e}\n"
            w += "(10% of mean y_obs magnitude). We highly recommend you\n"
            w += "explicitly set your estimate for the uncertainty on each\n"
            w += "observation.\n"
            warnings.warn(w) 
        
    def fit(self,
            model=None,
            guesses=None,
            names=None,
            lower_bounds=None,
            upper_bounds=None,
            prior_means=None,
            prior_stds=None,
            fixed=None,
            y_obs=None,
            y_std=None,
            **kwargs):
        """
        Fit the parameters.

        Parameters
        ----------

        model : callable
            model to fit.  Model should take "guesses" as its only argument.
            If model is a ModelWrapper instance, arguments related to the
            parameters (guess, bounds, names) will automatically be
            filled in.
        guesses : array of floats
            guesses for parameters to be optimized.
        y_obs : array of floats
            observations in an concatenated array
        bounds : list, optional
            list of two lists containing lower and upper bounds.  If None,
            bounds are set to -np.inf and np.inf
        priors : list, optional
            list of two lists containing the mean and standard deviation of 
            gaussian priors. None entries use uniform priors. If whole argument
            is None, use uniform priors for all parameters. 
        names : array of str
            names of parameters.  If None, parameters assigned names p0,p1,..pN
        y_std : array of floats or None
            standard deviation of each observation.  if None, each observation
            is assigned an error of 1.
        **kwargs : any remaining keyword arguments are passed as **kwargs to
            core engine (optimize.least_squares or emcee.EnsembleSampler)
        """

        # Make sure model already exists or create one based on existing 
        # self.model and model, guesses, names arguments. 
        self._process_model_args(model=model,
                                 guesses=guesses,
                                 names=names)

        # load guesses etc. into param_df
        self._process_fit_args(guesses=guesses,
                               lower_bounds=lower_bounds,
                               upper_bounds=upper_bounds,
                               prior_means=prior_means,
                               prior_stds=prior_stds,
                               fixed=fixed)
        
        # Load y_obs and y_std into attributes
        self._process_obs_args(y_obs=y_obs,
                               y_std=y_std)

        # Final check that everything is loaded 
        self._sanity_check("fit can be done",["model","y_obs","y_std"])
        
        # No fit has been run
        self._success = None

        # Finalize model
        self._model.finalize_params()

        # Run the fit
        self._fit(**kwargs)

        self._fit_has_been_run = True

    def _fit(self,**kwargs):
        """
        Should be redefined in subclass. This function should: 

        1. Update self._fit_result in whatever way makes sense for the fit. 
        2. Update self._success with True or False, depending on success. 
        3. Call self._update_fit_df
        """

        raise NotImplementedError("should be implemented in subclass\n")


    def _unweighted_residuals(self,param):
        """
        Private function calculating residuals with no error checking. 

        Parameters
        ----------
        param : numpy.ndarray
            float array of parameters

        Returns
        -------
        residuals : numpy.ndarray
            difference between observed and calculated values
        """

        y_calc = self.model(param)
        return self._y_obs - y_calc

    def unweighted_residuals(self,param):
        """
        Calculate residuals: (y_obs - y_calc)

        Parameters
        ----------
        param : numpy.ndarray
            float array of parameters

        Returns
        -------
        residuals : numpy.ndarray
            difference between observed and calculated values
        """

        self._sanity_check("fit can be done",["model","y_obs"])

        param = check_array(value=param,
                            variable_name="param",
                            expected_shape=(self.num_params,),
                            expected_shape_names="(num_param,)")
        
        return self._unweighted_residuals(param)

    def _weighted_residuals(self,param):
        """
        Private function calculating weighted residuals without error checking

        Parameters
        ----------
        param : numpy.ndarray
            float array of parameters

        Returns
        -------
        residuals : numpy.ndarray
            difference between observed and calculated values weighted by
            standard deviation
        """

        y_calc = self.model(param)
        return (self._y_obs - y_calc)/self._y_std

    def weighted_residuals(self,param):
        """
        Calculate weighted residuals: (y_obs - y_calc)/y_std

        Parameters
        ----------
        param : numpy.ndarray
            float array of parameters

        Returns
        -------
        residuals : numpy.ndarray
            difference between observed and calculated values weighted by
            standard deviation
        """

        self._sanity_check("fit can be done",["model","y_obs","y_std"])

        param = check_array(value=param,
                            variable_name="param",
                            expected_shape=(self.num_params,),
                            expected_shape_names="(num_param,)")
        
        return self._weighted_residuals(param)

    def _ln_like(self,param):
        """
        Private log likelihood, no error checking.

        Parameters
        ----------
        param : numpy.ndarray
            float array of parameters

        Returns
        -------
        ln_like : numpy.ndarray
            log likelihood 
        """

        y_calc = self.model(param)
        sigma2 = self._y_std**2
        return -0.5*(np.sum((self._y_obs - y_calc)**2/sigma2 + np.log(sigma2)))

    def ln_like(self,param):
        """
        Log likelihood: P(obs|model(param))

        Parameters
        ----------
        param : numpy.ndarray
            float array of parameters

        Returns
        -------
        ln_like : numpy.ndarray
            log likelihood
        """

        self._sanity_check("fit can be done",["model","y_obs","y_std",])

        param = check_array(value=param,
                            variable_name="param",
                            expected_shape=(self.num_params,),
                            expected_shape_names="(num_param,)")
        
        return self._ln_like(param)

        
    @property
    def model(self):
        """
        Model to use for calculating y_calc. The model should either be an 
        instance of dataprob.ModelWrapper OR a function that takes param (a 1D
        numpy array) as its only argument and returns y_calc (a 1D numpy array)
        as its only returned value. 
        """

        try:
            return self._model.model
        except AttributeError:
            return None

    @model.setter
    def model(self,model):

        if hasattr(self,"_model"):
            err = "Cannot add a new model because the model is already defined\n"
            raise ValueError(err)

        if not issubclass(type(model),ModelWrapper):
            err = "model must be a ModelWrapper instance\n"
            raise ValueError(err)
        
        self._model = model
        self._fit_has_been_run = False

        # Initialize the fit df now that we have a model
        self._initialize_fit_df()

    @property
    def y_obs(self):
        """
        Observed y values for fit. This should be a 1D numpy array of floats 
        the same length as the number of observations. 
        """

        try:
            return self._y_obs
        except AttributeError:
            return None

    @y_obs.setter
    def y_obs(self,y_obs):
        
        y_obs = check_array(value=y_obs,
                            variable_name="y_obs",
                            expected_shape=(None,),
                            expected_shape_names="(num_obs,)",
                            nan_allowed=False)
        self._y_obs = y_obs
        self._fit_has_been_run = False

    @property
    def y_std(self):
        """
        Estimated standard deviation on observed y values. This should be a 1D
        numpy array of floats the same length as the number of observations. 
        """

        try:
            return self._y_std
        except AttributeError:
            return None

    @y_std.setter
    def y_std(self,y_std):

        # Make sure the number of observations is known (determined by 
        # self._y_obs, ultimately)
        if self.num_obs is None:
            err = "y_std cannot be specified before y_obs\n"
            raise ValueError(err)
            
        # If the user sends in a single value, try to expand to an array 
        if not hasattr(y_std,"__iter__"):

            # Make sure it's a float
            y_std = check_float(value=y_std,
                                variable_name="y_std")

            # Expanded y_std
            y_std = y_std*np.ones(self.num_obs,dtype=float)
            
        # Make sure array has correct dimensions
        y_std = check_array(value=y_std,
                            variable_name="y_std",
                            expected_shape=(self.num_obs,),
                            expected_shape_names=f"({self.num_obs},)",
                            nan_allowed=False)
        
        # no values < 0 allowed
        if np.sum(y_std < 0) > 0:
            err = "all entries in y_std must be greater than zero\n"
            raise ValueError(err)
        
        self._y_std = y_std
        self._fit_has_been_run = False


    @property
    def param_df(self):
        """
        Return a dataframe with fit parameters. 
        """

        try:
            return self._model.param_df
        except AttributeError:
            return None
        
    @property
    def data_df(self):

        out = {}
        
        y_obs = self.y_obs
        if y_obs is not None:
            out["y_obs"] = y_obs

        y_std = self.y_std
        if y_std is not None:
            out["y_std"] = y_std

        if self.success:
            estimate = np.array(self.fit_df["estimate"])
            out["y_calc"] = self.model(estimate)
            out["unweighted_residuals"] = self._unweighted_residuals(estimate)
            out["weighted_residuals"] = self._weighted_residuals(estimate)

        return pd.DataFrame(out)

    def _initialize_fit_df(self):

        df = pd.DataFrame({"name":self.param_df["name"]})
        df.index = df["name"]
        df["estimate"] = np.nan
        df["std"] = np.nan
        df["low_95"] = np.nan
        df["high_95"] = np.nan

        self._fit_df = df

    def _update_fit_df(self):
        """
        Should be redefined in subclass. This function should update 
        self._fit_df. 
        """

        raise NotImplementedError("should be implemented in subclass\n")

    @property
    def fit_df(self):
        """
        Return the fit results as a dataframe.
        """

        try:
            return self._fit_df
        except AttributeError:
            return None

    @property
    def samples(self):
        """
        Samples of fit parameters. If fit has been run and generated samples, 
        this will be a float numpy array with a shape (num_samples,num_param). 
        Otherwise, is None. 
        """

        try:
            return self._samples
        except AttributeError:
            return None

    def write_samples(self,output_file):
        """
        Write the samples from the fit out to a pickle file.

        Parameters
        ----------
        output_file : str
            output pickle file to write to
        """

        # See if the file exists already.
        if os.path.isfile(output_file):
            err = f"{output_file} exists.\n"
            raise FileExistsError(err)

        # If there are samples, write them out.
        if self.samples is not None:
            with open(output_file,"wb") as p:
                pickle.dump(self.samples,p)

    def append_samples(self,sample_file=None,sample_array=None):
        """
        Append samples to the fit.  The new samples must be a float array
        with the shape: num_samples, num_parameters. This can come from a
        pickle file (sample_file) or array.  Only one of these can be specified.

        Parameters
        ----------
        sample_file : str
            Pickle file of the numpy array
        sample_array : numpy.ndarray
            array of samples
        """

        # Nothing to do; no new samples specified
        if sample_file is None and sample_array is None:
            return

        # Samples must already exist to append
        if self.samples is None:
            err = "You can only append samples to a fit that has already been done.\n"
            raise ValueError(err)

        # Cannot specify both sample file and sample array
        if sample_file is not None and sample_array is not None:
            err = "Either a file with samples or a sample array can be specified,"
            err += " but not both."
            raise ValueError(err)

        # Read sample file
        if sample_file is not None:

            try:
                sample_array = pickle.load(open(sample_file,"rb"))
            except FileNotFoundError:
                err = f"'{sample_file}'  does not exist.\n"
                raise FileNotFoundError(err)
            except pickle.UnpicklingError:
                err = f"'{sample_file}' does not appear to be a pickle file.\n"
                raise pickle.UnpicklingError(err)

        # Check sanity of sample_array; update num_params if not specified
        try:
            sample_array = np.array(sample_array,dtype=float)
        except Exception as e:
            err = "sample_array should be a float numpy array\n"
            raise ValueError(err) from e
        
        if len(sample_array.shape) != 2:
            err = "sample_array should have dimensions (num_samples,num_param)\n"
            raise ValueError(err)

        if sample_array.shape[1] != self.num_params:
            err = "sample_array should have dimensions (num_samples,num_param)\n"
            raise ValueError(err)

        # Concatenate the new samples to the existing samples
        self._samples = np.concatenate((self.samples,sample_array))

        self._update_fit_df()

    def get_sample_df(self,num_samples=100):
        """
        Create a dataframe with y_calc for samples from parameter uncertainty as
        columns. The output dataframe will have the columns 'y_obs', 'y_std',
        'y_calc', 's0', 's1', ... 'sn', where 'y_calc' is the calculated value
        using the ``self.fit_df["estimate"]`` parameters and s0 through sn are
        calculated values using parameters sampled from the estimated 
        likelihood surface. If no samples have been generated, the dataframe
        will omit the s0-sn columns. 

        Parameters
        ----------
        num_samples : int
            number of samples to take. 

        Returns
        -------
        sample_df : pandas.DataFrame
            dataframe with y_obs, y_calc, and values sampled from likelihood
            surface. 
        """

        num_samples = check_int(value=num_samples,
                                variable_name="num_samples",
                                minimum_allowed=0)

        # Out dictionary
        out = {}
        
        # Get y_obs if defined
        y_obs = self.y_obs
        if y_obs is not None:
            out["y_obs"] = y_obs

        # get y_std if defined
        y_std = self.y_std
        if y_std is not None:
            out["y_std"] = y_std

        # get y_calc if fit was successful
        if self.success:
            out["y_calc"] = self.model(self.fit_df["estimate"])

        samples = self.samples
        if samples is not None:
            
            N = samples.shape[0]
            fmt_string = _pretty_zeropad_str(N)

            for i in range(0,N,N//(num_samples-1)):
                key = fmt_string.format(i)
                out[key] = self.model(self.samples[i])
    
        return pd.DataFrame(out)

    def corner_plot(self,filter_params=None,**kwargs):
        """
        Create a "corner plot" that shows distributions and correlations of 
        values for all fit parameters. This can only be run if the analysis 
        has generated samples. 

        Parameters
        ----------
        filter_params : list-like, optional
            strings used to search parameter names.  If a parameter name matches
            one of the patterns, it is *not* plotted. 
        **kwargs : 
            any extra keyword arguments are passed directly to corner.corner 
            to tune formatting, etc. To learn more, ``import corner`` then
            ``help(corner.corner)``. 

        Returns
        -------
        fig : matplotlib.Figure
            matplotlib figure instance generated by calling corner.corner
        """

        # Don't return anything if this is the base class
        if self.fit_type == "":
            return None

        # if filter parameters are not specified, no skip_pattern
        if filter_params is None:
            skip_pattern = None

        # otherwise
        else:

            # If the user passes a string (instead of a list or tuple of patterns),
            # convert it to a list up front.
            if type(filter_params) is str:
                filter_params = (filter_params,)

            # Make sure it's strings
            filter_params = [str(p) for p in filter_params]

            # compile a pattern to look for
            skip_pattern = re.compile("|".join(filter_params))

        # Check for samples
        if self.samples is None:
            err = "Fit does not have samples. Could not generate a corner plot.\n"
            raise RuntimeError(err)
        
        # Go through samples
        keep_indexes = []
        corner_range = []
        names = []
        est_values = []
        for i in range(self.samples.shape[1]):

            # Get name and estimate
            name = self.fit_df["name"][i]
            estimate = self.fit_df["estimate"][i]

            # look for patterns to skip
            if skip_pattern is not None and skip_pattern.search(name):
                print("not doing corner plot for parameter ",name)
                continue

            names.append(name)
            keep_indexes.append(i)
            corner_range.append(tuple([np.min(self.samples[:,i])-0.5,
                                       np.max(self.samples[:,i])+0.5]))
            est_values.append(estimate)

        # make sure we kept at least one parameter
        if len(keep_indexes) == 0:
            err = "filter_params removed all parameters. Could not generate\n"
            err += "corner plot\n"
            raise ValueError(err)

        # Create array to plot samples
        to_plot = self.samples[:,np.array(keep_indexes,dtype=int)]

        # Load labels, range, and truths into kwargs only if the user has not
        # defined them as explicit kwargs. User corner.corner to check sanity 
        # of their inputs. 
        if "labels" not in kwargs:
            kwargs["labels"] = names
        if "range" not in kwargs:
            kwargs["range"] = corner_range
        if "truths" not in kwargs:
            kwargs["truths"] = est_values

        # Call corner 
        fig = corner.corner(to_plot,**kwargs)

        return fig

    @property
    def num_params(self):
        """
        Number of fit parameters. If model has not been defined, will be None. 
        """

        try:
            return len(self._model.param_df)
        except AttributeError:
            return None

    @property
    def num_obs(self):
        """
        Number of observations. If y_obs has not been defined, will be None. 
        """

        try:
            return self._y_obs.shape[0]
        except AttributeError:
            return None

    @property
    def fit_type(self):
        """
        Fit type as a string. 
        """
        return self._fit_type

    @property
    def success(self):
        """
        Whether the fit was successful when run (True or False). If no fit has
        been attempted, will be None.
        """

        try:
            return self._success
        except AttributeError:
            return None

    @property
    def fit_info(self):
        """
        Should be implemented in subclass. Information about fit run as a 
        dictionary. 
        """

        raise NotImplementedError("should be implemented in subclass\n")
    
    @property
    def fit_result(self):
        """
        Full fit results (will depend on exact fit type what is placed here).
        """

        try:
            return self._fit_result
        except AttributeError:
            return None