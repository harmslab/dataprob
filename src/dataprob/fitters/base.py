"""
Fitter base class allowing different classes of fits.
"""

import numpy as np
import pandas as pd
import corner


import re
import inspect
import pickle
import os
import warnings

from dataprob.check import check_array
from dataprob.model_wrapper.model_wrapper import ModelWrapper
    

class Fitter:
    """
    Base class for fits/analyses using a dataprob function.
    """

    def __init__(self):
        """
        Init function for the class.
        """

        self._num_obs = None
        self._num_params = None

        # Keep track of whether or not the model is a ModelWrapper instance
        # (which can be used for more powerful fit/parameter options)
        self._model_is_model_wrapper = False

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

    def fit(self,
            model=None,
            guesses=None,
            y_obs=None,
            bounds=None,
            priors=None,
            names=None,
            y_stdev=None,
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
        y_stdev : array of floats or None
            standard deviation of each observation.  if None, each observation
            is assigned an error of 1.
        **kwargs : any remaining keyword arguments are passed as **kwargs to
            core engine (optimize.least_squares or emcee.EnsembleSampler)
        """

        # Record model, check for preloaded model, or fail.
        if model is not None:
            self.model = model
        else:
            if self.model is None:
                err = "model must be specified before fit\n"
                raise RuntimeError(err)

        # Record guesses, grab from ModelWrapper model, or fail.
        if guesses is not None:
            self.guesses = guesses
        else:
            if self.guesses is None:
                err = "parameter guesses must be specified before fit\n"
                raise RuntimeError(err)

        # Record bounds, grab from ModelWrapper model, or make infinite
        if bounds is not None:
            self.bounds = bounds
        else:
            if self.bounds is None:
                tmp = np.ones(len(self.guesses))
                self.bounds = [-np.inf*tmp,np.inf*tmp]

        # Record priors, grab from ModelWrapper model, or make nan (uniform)
        if priors is not None:
            self.priors = priors
        else:
            if self.priors is None:
                self.priors = np.nan*np.ones((2,len(self.guesses)),
                                                dtype=float)

        # Record names, grab from ModelWrapper model, or use default
        if names is not None:
            self.names = names
        else:
            if self.names is None:
                self.names = ["p{}".format(i) for i in range(len(self.guesses))]

        # Record y_obs, check for preloaded, or fail
        if y_obs is not None:
            self.y_obs = y_obs
        else:
            if self.y_obs is None:
                err = "y_obs must be specified before fit\n"
                raise RuntimeError(err)

        # Record y_stdev, check for preloaded, or use default
        if y_stdev is not None:
            self.y_stdev = y_stdev
        else:
            if self.y_stdev is None:

                scalar = np.mean(np.abs(self.y_obs))*0.1
                self.y_stdev = scalar*np.ones(len(self.y_obs),dtype=float)

                w = "\ny_stdev must be sent in for proper residual/likelihood\n"
                w += f"calculation. We have arbitrarily set it to {scalar:.2e}\n"
                w += "(10% of mean y_obs magnitude). We highly recommend you\n"
                w += "explicitly set your estimate for the uncertainty on each\n"
                w += "observation.\n"
                warnings.warn(w) 

        # No fit has been run
        self._success = None

        self._sanity_check("fit can be done",["model","y_obs","y_stdev"])

        # Make sure that there is at least one adjustable parameter
        if len(self.guesses) < 1:
            err = "Cannot do fit with no adjustable parameters\n"
            raise RuntimeError(err)

        self._fit(**kwargs)

        # Load the fit results into the model_wrapper
        if self._model_is_model_wrapper:
            self._model.load_fit_result(self)

        self._fit_has_been_run = True

    def _fit(self,**kwargs):
        """
        Should be redefined in subclass.
        """

        raise NotImplementedError("should be implemented in subclass\n")

    def _update_estimates(self):
        """
        Should be redefined in subclass.
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
        
        # This call should finalize the number of parameters if not already set
        if self.num_params is None:
            self._num_params = len(param)

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
        return (self.y_obs - y_calc)/self.y_stdev

    def weighted_residuals(self,param):
        """
        Calculate weighted residuals: (y_obs - y_calc)/y_stdev

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

        self._sanity_check("fit can be done",["model","y_obs","y_stdev"])

        param = check_array(value=param,
                            variable_name="param",
                            expected_shape=(self.num_params,),
                            expected_shape_names="(num_param,)")
        
        # This call should finalize the number of parameters if not already set
        if self.num_params is None:
            self._num_params = len(param)

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
        sigma2 = self._y_stdev**2
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

        self._sanity_check("fit can be done",["model","y_obs","y_stdev",])

        param = check_array(value=param,
                            variable_name="param",
                            expected_shape=(self.num_params,),
                            expected_shape_names="(num_param,)")
        
        # This call should finalize the number of parameters if not already set
        if self.num_params is None:
            self._num_params = len(param)

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
            if self._model_is_model_wrapper:
                return self._model.model
            else:
                return self._model
        except AttributeError:
            return None

    @model.setter
    def model(self,model):

        # User sent in ModelWrapper instance, not method. model_to_check should be
        # the method; model should be the class. 
        if issubclass(type(model),ModelWrapper):
            model_is_model_wrapper = True
            model = model
            model_to_check = model.model

        # User set in ModelWrapper method, not instance. model_to_check should be 
        # the method (model); model should be the class (model.__self__)
        elif hasattr(model,"__self__") and issubclass(type(model.__self__),ModelWrapper):
            model_is_model_wrapper = True
            model_to_check = model
            model = model.__self__
            
        # Not a model wrapper. Just treat as a class. model_to_check and model 
        # should be the same. 
        else:
            model_is_model_wrapper = False
            model = model
            model_to_check = model

        # Make sure it's callable
        if not hasattr(model_to_check,"__call__"):
            err = "model must be a function that takes at least one argument\n"
            raise ValueError(err)

        # Make sure it takes at least one parameter
        if len(inspect.signature(model_to_check).parameters) < 1:
            err = "model must be a function that takes at least one argument\n"
            raise ValueError(err)
        
        # Make sure the number of parameters match between the model and the 
        # fitter.
        if model_is_model_wrapper and self._num_params is not None:

            if len(model.names) != self._num_params:
                err = f"number of model parameters ({len(model.names)}) does\n"
                err += f"not match the number of parameters in the Fitter ({self._num_params})\n"
                raise ValueError(err)

        # Set attributes -- we passed all tests
        self._model_is_model_wrapper = model_is_model_wrapper
        self._model = model
        self._fit_has_been_run = False

    @property
    def names(self):
        """
        Parameter names for fit parameters.

        Should be an array of unique strings the same length as the number of
        parameters. 
        """

        # Grab the bounds from the model wrapper in case they changed
        if self._model_is_model_wrapper:
            self._names = self._model.names

        try:
            return self._names
        except AttributeError:
            return None

    @names.setter
    def names(self,names):

        if self._model_is_model_wrapper:
            err = "parameter names cannot be set when using a ModelWrapper.\n"
            err += "These can only be set when doing the initial wrapping.\n"
            raise RuntimeError(err)

        # If the user sends in a naked string, make it into a list of strings
        if issubclass(type(names),str):
            names = [names]

        # Force to be an array of strings
        names = np.array(names,dtype=str)
        
        if len(names) != len(set(names)):
            doc = inspect.getdoc(Fitter.names)
            err = f"parameter names must all be unique. \n\n{doc}\n\n"
            raise ValueError(err)

        if self.num_params is not None:
            if names.shape[0] != self.num_params:
                doc = inspect.getdoc(Fitter.names)
                err = f"length of names ({names.shape[0]}) must match the\n"
                err += f"number of parameters ({self.num_params}) \n\n{doc}\n\n"
                raise ValueError(err)
        else:
            self._num_params = names.shape[0]

        self._names = names

    @property
    def guesses(self):
        """
        Guesses for fit parameters.
        """

        # Grab the guesses from the model wrapper in case they changed
        if self._model_is_model_wrapper:
            self._guesses = self._model.guesses

        try:
            return self._guesses
        except AttributeError:
            return None

    @guesses.setter
    def guesses(self,guesses):

        guesses = check_array(value=guesses,
                              variable_name="guesses",
                              expected_shape=(None,),
                              expected_shape_names="(num_param,)")
        
        if self.num_params is None:
            self._num_params = guesses.shape[0]

        if guesses.shape[0] != self.num_params:
            err = "guesses should be a numpy array the same length as the\n"
            err += "number of guesses\n"
            raise ValueError(err)
        
        self._guesses = guesses

        # Update the underlying guesses in each FitParameter instance
        if self._model_is_model_wrapper:
            self._model.guesses = guesses

        self._fit_has_been_run = False

    @property
    def bounds(self):
        """
        Bounds for fit parameters.

        bounds must be a (2 x num_parameters) numpy array of floats with the
        form:

        [[lower_0, lower_1, ..., lower_n],
         [upper_0, upper_1, ..., upper_n]]

        np.inf values are allowed, indicating no bounds on that parameter.
        np.nan are not allowed. 
        """

        # Grab the bounds from the model wrapper in case they changed
        if self._model_is_model_wrapper:
            self._bounds = self._model.bounds

        try:
            return self._bounds
        except AttributeError:
            return None

    @bounds.setter
    def bounds(self,bounds):

        bounds = check_array(value=bounds,
                              variable_name="bounds",
                              expected_shape=(2,None),
                              expected_shape_names="(2,num_param)")
        
        if self.num_params is None:
            self._num_params = bounds.shape[1]

        if bounds.shape[1] != self.num_params:
            doc = inspect.getdoc(Fitter.bounds)
            err = f"incorrectly specified bounds. \n\n{doc}\n\n"
            raise ValueError(err)

        self._bounds = bounds

        # Update the underlying guesses in each FitParameter instance
        if self._model_is_model_wrapper:
            self._model.bounds = bounds

        self._fit_has_been_run = False

    @property
    def priors(self):
        """
        Gaussian priors to use for each parameter.

        priors must be a (2 x num_parameters) numpy array of floats with the
        form:

        [[mean_0,   mean_1, ...,  mean_n],
         [stdev_0, stdev_1, ..., stdev_n]]

        np.inf values are not allowed. np.nan entries indicate that uniform
        priors should be used for that parameter.        
        """

        # Grab the priors from the model wrapper in case they changed
        if self._model_is_model_wrapper:
            self._priors = self._model.priors

        try:
            return self._priors
        except AttributeError:
            return None

    @priors.setter
    def priors(self,priors):

        priors = check_array(value=priors,
                             variable_name="priors",
                             expected_shape=(2,None),
                             expected_shape_names="(2,num_param)")
        
        if self.num_params is None:
            self._num_params = priors.shape[1]

        if priors.shape[1] != self.num_params:
            doc = inspect.getdoc(Fitter.priors)
            err = f"incorrectly specified priors. \n\n{doc}\n\n"
            raise ValueError(err)

        if np.sum(np.isinf(priors)) > 0:
            doc = inspect.getdoc(Fitter.priors)
            err = f"priors cannot be infinite. \n\n{doc}\n\n"
            raise ValueError(err)

        self._priors = priors

        # Update the underlying guesses in each FitParameter instance
        if self._model_is_model_wrapper:
            self._model.priors = priors

        self._fit_has_been_run = False

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
                            expected_shape_names="(num_obs,)")
        
        if self.num_obs is None:
            self._num_obs = y_obs.shape[0]

        if y_obs.shape[0] != self.num_obs:
            doc = inspect.getdoc(Fitter.y_obs)
            err = f"incorrectly specified y_obs. \n\n{doc}\n\n"
            raise ValueError(err)

        self._y_obs = y_obs

        self._fit_has_been_run = False

    @property
    def y_stdev(self):
        """
        Estimated standard deviation on observed y values. This should be a 1D
        numpy array of floats the same length as the number of observations. 
        """

        try:
            return self._y_stdev
        except AttributeError:
            return None

    @y_stdev.setter
    def y_stdev(self,y_stdev):

        y_stdev = check_array(value=y_stdev,
                              variable_name="y_stdev",
                              expected_shape=(None,),
                              expected_shape_names="(num_obs,)")
        
        if self.num_obs is None:
            self._num_obs = y_stdev.shape[0]

        if y_stdev.shape[0] != self.num_obs:
            doc = inspect.getdoc(Fitter.y_stdev)
            err = f"incorrectly specified y_stdev. \n\n{doc}\n\n"
            raise ValueError(err)

        self._y_stdev = y_stdev

        self._fit_has_been_run = False

    @property
    def num_params(self):
        """
        Number of fit parameters.
        """

        if self._model_is_model_wrapper:
            return len(self._model.names)

        try:
            return self._num_params
        except AttributeError:
            return None

    @property
    def num_obs(self):
        """
        Number of observations.
        """

        try:
            return self._num_obs
        except AttributeError:
            return None

    @property
    def estimate(self):
        """
        Estimates of fit parameters.
        """

        try:
            return self._estimate
        except AttributeError:
            return None

    @property
    def stdev(self):
        """
        Standard deviations on estimates of fit parameters.
        """

        try:
            return self._stdev
        except AttributeError:
            return None

    @property
    def ninetyfive(self):
        """
        Ninety-five perecent confidence intervals on the estimates.
        """

        try:
            return self._ninetyfive
        except AttributeError:
            return None

    @property
    def fit_result(self):
        """
        Full fit results (will depend on exact fit type what is placed here).
        """

        try:
            return self._fit_result
        except AttributeError:
            return None

    @property
    def success(self):
        """
        Whether the fit was successful.
        """

        try:
            return self._success
        except AttributeError:
            return None

    @property
    def fit_info(self):
        """
        Information about fit run.
        """

        return None

    @property
    def samples(self):
        """
        Samples of fit parameters.
        """

        try:
            return self._samples
        except AttributeError:
            return None

    @property
    def fit_type(self):
        """
        Fit type. 
        """
        return self._fit_type

    @property
    def fit_df(self):
        """
        Return the fit results as a dataframe.
        """

        if not self.success:
            return None

        out_dict = {"param":[],
                    "estimate":[],
                    "stdev":[],
                    "low_95":[],
                    "high_95":[],
                    "guess":[],
                    "prior_mean":[],
                    "prior_std":[],
                    "lower_bound":[],
                    "upper_bound":[]}

        if self._model_is_model_wrapper:

            m = self._model

            out_dict["fixed"] = []
            for p in m.fit_parameters.keys():
                out_dict["param"].append(p)
                out_dict["estimate"].append(m.fit_parameters[p].value)
                out_dict["fixed"].append(m.fit_parameters[p].fixed)

                if m.fit_parameters[p].fixed:
                    for col in ["stdev","low_95","high_95","guess",
                                "lower_bound","upper_bound",
                                "prior_mean","prior_std"]:
                        out_dict[col].append(None)
                else:
                    out_dict["stdev"].append(m.fit_parameters[p].stdev)

                    if m.fit_parameters[p].ninetyfive is not None:
                        out_dict["low_95"].append(m.fit_parameters[p].ninetyfive[0])
                        out_dict["high_95"].append(m.fit_parameters[p].ninetyfive[1])
                    else:
                        out_dict["low_95"].append(np.nan)
                        out_dict["high_95"].append(np.nan)

                    out_dict["guess"].append(m.fit_parameters[p].guess)
                    out_dict["lower_bound"].append(m.fit_parameters[p].bounds[0])
                    out_dict["upper_bound"].append(m.fit_parameters[p].bounds[1])
                    out_dict["prior_mean"].append(m.fit_parameters[p].prior[0])
                    out_dict["prior_std"].append(m.fit_parameters[p].prior[1])

        else:

            for i in range(len(self.names)):

                out_dict["param"].append(self.names[i])
                out_dict["estimate"].append(self.estimate[i])
                out_dict["stdev"].append(self.stdev[i])

                if self.ninetyfive is not None:
                    out_dict["low_95"].append(self.ninetyfive[0,i])
                    out_dict["high_95"].append(self.ninetyfive[1,i])
                else:
                    out_dict["low_95"].append(np.nan)
                    out_dict["high_95"].append(np.nan)

                out_dict["guess"].append(self.guesses[i])
                out_dict["lower_bound"].append(self.bounds[0,i])
                out_dict["upper_bound"].append(self.bounds[1,i])
                out_dict["prior_mean"].append(self.priors[0,i])
                out_dict["prior_std"].append(self.priors[1,i])

        return pd.DataFrame(out_dict)

    @property
    def data_df(self):

        out = {}
        
        y_obs = self.y_obs
        if y_obs is not None:
            out["y_obs"] = y_obs

        y_stdev = self.y_stdev
        if y_stdev is not None:
            out["y_stdev"] = y_stdev

        estimate = self.estimate
        if estimate is not None:
            out["y_calc"] = self.model(estimate)
            out["unweighted_residuals"] = self._unweighted_residuals(estimate)
            out["weighted_residuals"] = self._weighted_residuals(estimate)

        return pd.DataFrame(out)

    
    def get_sample_df(self,num_samples=100):

        out = {}
        
        y_obs = self.y_obs
        if y_obs is not None:
            out["y_obs"] = y_obs

        y_stdev = self.y_stdev
        if y_stdev is not None:
            out["y_stdev"] = y_stdev

        estimate = self.estimate
        if estimate is not None:
            out["y_calc"] = self.model(estimate)

        samples = self.samples
        if samples is not None:
            
            N = samples.shape[0]
            num_digits = len(f"{N}") + 1
            fmt_string = "s{:0" + f"{num_digits}" + "d}"
            for i in range(0,N,N//(num_samples-1)):
                key = fmt_string.format(i)
                out[key] = self.model(self.samples[i])
    
        return pd.DataFrame(out)

            

        


    def corner_plot(self,filter_params=("DUMMY_FILTER",),*args,**kwargs):
        """
        Create a "corner plot" that shows distributions of values for each
        parameter, as well as cross-correlations between parameters.

        Parameters
        ----------
        filter_params : list-like
            strings used to search parameter names.  if the string matches,
            the parameter is *excluded* from the plot.
        """

        # Don't return anything if this is the base class
        if self.fit_type == "":
            return None

        # If the user passes a string (instead of a list or tuple of patterns),
        # convert it to a list up front.
        if type(filter_params) is str:
            filter_params = (filter_params,)

        skip_pattern = re.compile("|".join(filter_params))

        s = self.samples

        # Make sure that fit actually returned samples. (Will fail, for example
        # if Jacobian misbehaves in ML fit)
        if len(s) == 0:
            err = "\n\nFit did not produce samples for generation of a corner plot.\nCheck warnings.\n"
            raise RuntimeError(err)

        keep_indexes = []
        corner_range = []
        names = []
        est_values = []
        for i in range(s.shape[1]):

            # look for patterns to skip
            if skip_pattern.search(self.names[i]):
                print("not doing corner plot for parameter ",self.names[i])
                continue

            names.append(self.names[i])
            keep_indexes.append(i)
            corner_range.append(tuple([np.min(s[:,i])-0.5,np.max(s[:,i])+0.5]))

            est_values.append(self.estimate[i])

        to_plot = s[:,np.array(keep_indexes,dtype=int)]

        fig = corner.corner(to_plot,labels=names,range=corner_range,
                            truths=est_values,*args,**kwargs)

        return fig

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

        self._update_estimates()
