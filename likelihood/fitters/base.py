__description__ = \
"""
Fitter base class allowing different classes of fits.
"""
__author__ = "Michael J. Harms"
__date__ = "2017-05-10"

import numpy as np
import scipy.stats
import scipy.optimize as optimize
import corner
import pandas as pd

import re, inspect, pickle, os, warnings

import likelihood

class Fitter:
    """
    Base class for fits/analyses using a likelihood function.
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

        self.fit_type = ""

    def _sanity_check(self,call_string,attributes_to_check):
        """
        Do a sanity check before doing model calculations.

        call_string: string to dump into output on error.
        """

        for a in attributes_to_check:
            try:
                if self.__dict__[f"_{a}"] is None:
                    raise KeyError
            except KeyError:
                err = f"'{a}' must be set before {call_string}\n"
                raise RuntimeError(err)

    def fit(self,model=None,guesses=None,y_obs=None,bounds=None,names=None,y_stdev=None,**kwargs):
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
        bounds : list
            list of two lists containing lower and upper bounds.  If None,
            bounds are set to -np.inf and np.inf
        names : array of str
            names of parameters.  If None, parameters assigned names p0,p1,..pN
        y_stdev : array of floats or None
            standard deviation of each observation.  if None, each observation
            is assigned an error of 1.
        **kwargs : any remaining keywaord arguments are passed as **kwargs to
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
                if self._model_is_model_wrapper:
                    self.guesses = self.model.guesses
                else:
                    err = "parameter guesses must be specified before fit\n"
                    raise RuntimeError(err)

        # Record bounds, grab from ModelWrapper model, or make infinite
        if bounds is not None:
            self.bounds = bounds
        else:
            if self.bounds is None:
                if self._model_is_model_wrapper:
                    self.bounds = self.model.bounds
                else:
                    tmp = np.ones(len(self.guesses))
                    self.bounds = [-np.inf*tmp,np.inf*tmp]

        # Record names, grab from ModelWrapper model, or use default
        if names is not None:
            self.names = names
        else:
            if self.names is None:
                if self._model_is_model_wrapper:
                    self.names = self.model.names
                else:
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
                self.y_stdev = np.ones(len(self.y_obs),dtype=np.float)

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

        pass

    def _update_estimates(self):
        """
        Should be redefined in subclass.
        """

        pass


    @property
    def fit_to_df(self):
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
                    "lower_bound":[],
                    "upper_bound":[]}

        if self._model_is_model_wrapper:

            m = self._model

            out_dict["fixed"] = []
            for p in m.fit_parameters.keys():
                out_dict["param"].append(m.fit_parameters[p].name)
                out_dict["estimate"].append(m.fit_parameters[p].value)
                out_dict["fixed"].append(m.fit_parameters[p].fixed)

                if m.fit_parameters[p].fixed:
                    for col in ["stdev","low_95","high_95","guess","lower_bound","upper_bound"]:
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

        return pd.DataFrame(out_dict)


    def unweighted_residuals(self,param):
        """
        Calculate residuals.
        """

        self._sanity_check("fit can be done",["model","y_obs"])

        y_calc = self.model(param)
        return self._y_obs - y_calc


    def weighted_residuals(self,param):
        """
        Calculate weighted residuals.
        """

        self._sanity_check("fit can be done",["model","y_obs","y_stdev"])

        y_calc = self.model(param)
        return (self.y_obs - y_calc)/self.y_stdev

    def ln_like(self,param):
        """
        Log likelihood of function given parameters.
        """

        self._sanity_check("fit can be done",["model","y_obs","y_stdev"])

        if self.model is not None:

            y_calc = self.model(param)
            sigma2 = self._y_stdev**2
            return -0.5*(np.sum((self._y_obs - y_calc)**2/sigma2 + np.log(sigma2)))
        else:
            return None

    @property
    def model(self):
        """
        Model used for fit.
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
        """
        Setter for "model" attribute.
        """

        if model is None:
            return

        # If this is a ModelWrapper instance, grab the model method rather than
        # the model instance for the check below.
        if isinstance(model,likelihood.model_wrapper.ModelWrapper):
            model = model.model

        has_err = False
        try:
            if not inspect.isfunction(model) and not inspect.ismethod(model):
                has_err = True
            if len(inspect.signature(model).parameters) < 1:
                has_err = True

        except TypeError:
            has_err = True

        if has_err:
            err = "model must be a function that takes at least one argument\n"
            raise ValueError(err)

        # If the model is a method of a ModelWrapper instance, record this so
        # the Fitter object knows it can get guesses, bounds, and names from
        # the ModelWrapper if necessary.
        self._model_is_model_wrapper = False
        try:
            if model.__qualname__.startswith("ModelWrapper"):

                # If this is a model wrapper, we can check for consistency
                # between the number of model parameters in the model vs.
                # what has been pre-set in the model.
                if self.num_params is not None:
                    if len(model.__self__.guesses) != self.num_params:
                        err = f"number of model parameters ({len(model.__self__.guesses)}) does\n"
                        err += f"not match the number of parameters in the Fitter ({self.num_params})\n"
                        raise ValueError(err)

                self._model_is_model_wrapper = True

                # We're going to store the ModelWrapper instance, not the
                # method.
                model = model.__self__

        except AttributeError:
            pass

        # Record the model
        self._model = model
        self._fit_has_been_run = False

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
        """
        Setter for guess attribute.
        """

        try:
            guesses = np.array(guesses,dtype=np.float)
        except (ValueError,TypeError) as err:
            err = f"{err}\n\nguesses must be a list or array of floats\n\n"
            raise ValueError(err)

        if self.num_params is not None:
            if guesses.shape[0] != self.num_params:
                err = "length of guesses ({}) must match the number of parameters ({})\n".format(guesses.shape[0],
                                                                                                 self.num_params)
                raise ValueError(err)
        else:
            self._num_params = guesses.shape[0]

        self._guesses = guesses

        # Update the underlying guesses in each FitParameter instance
        if self._model_is_model_wrapper:
            for i, p in enumerate(self._model.position_to_param):
                self._model.fit_parameters[p].guess = guesses[i]

        self._fit_has_been_run = False

    @property
    def bounds(self):
        """
        Bounds for fit parameters.
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
        """
        Setter for bounds attribute.
        """

        try:
            bounds = np.array(bounds,dtype=np.float)
            if len(bounds.shape) != 2 or bounds.shape[0] != 2:
                raise ValueError("incorrect dimensions!\n")
        except (ValueError,TypeError) as err:
            err = f"{err}\n\nguesses must be a 2 x num_params list or array of floats:\n\n"
            err += "   [[lower_1,lower_2,...lower_n],[upper_1,upper_2,...upper_n]]\n\n"
            raise ValueError(err)

        if self.num_params is not None:
            if bounds.shape[1] != self.num_params:
                err = "length of bounds ({}) must match the number of parameters ({})\n".format(bounds.shape[1],
                                                                                                self.num_params)
                raise ValueError(err)
        else:
            self._num_params = bounds.shape[1]

        self._bounds = bounds

        # Update the underlying guesses in each FitParameter instance
        if self._model_is_model_wrapper:
            for i, p in enumerate(self._model.position_to_param):
                self._model.fit_parameters[p].bounds = bounds[:,i]

        self._fit_has_been_run = False


    @property
    def names(self):
        """
        Parameter names for fit parameters.
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
        """
        Setter for parameter names attribute.
        """

        try:
            names = np.array(names,dtype=np.str)

            # Will throw a type error if the user puts in a single value
            try:
                len(names)
            except TypeError:
                names = np.array([names])

        except ValueError as err:
            err = f"{err}\n\nnames must be a list or array of strings\n\n"
            raise ValueError(err)

        if len(names) != len(set(names)):
            err = "parameter names must all be unique.\n"
            raise ValueError(err)

        if self.num_params is not None:
            if names.shape[0] != self.num_params:
                err = "length of names ({}) must match the number of parameters ({})\n".format(names.shape[0],
                                                                                                   self.num_params)
                raise ValueError(err)
        else:
            self._num_params = names.shape[0]

        self._names = names

        # Update the underlying guesses in each FitParameter instance
        if self._model_is_model_wrapper:
            for i, p in enumerate(self._model.position_to_param):
                self._model.fit_parameters[p].name = names[i]

    @property
    def y_obs(self):
        """
        Observed y values for fit.
        """

        try:
            return self._y_obs
        except AttributeError:
            return None

    @y_obs.setter
    def y_obs(self,y_obs):
        """
        Setter for y_obs attribute.
        """

        try:
            y_obs = np.array(y_obs,dtype=np.float)
        except (ValueError,TypeError) as err:
            err = f"{err}\n\ny_obs must be a list or array of floats\n\n"
            raise ValueError(err)

        if self._num_obs is not None:
            if y_obs.shape[0] != self._num_obs:
                err = "observation already loaded with a different size ({})\n".format(self._num_obs)
                raise ValueError(err)
        else:
            self._num_obs = y_obs.shape[0]

        self._y_obs = y_obs

        self._fit_has_been_run = False


    @property
    def y_stdev(self):
        """
        Observed y values for fit.
        """

        try:
            return self._y_stdev
        except AttributeError:
            return None

    @y_stdev.setter
    def y_stdev(self,y_stdev):
        """
        Setter for y_stdev attribute.
        """

        try:
            y_stdev = np.array(y_stdev,dtype=np.float)
        except (ValueError,TypeError) as err:
            err = f"{err}\n\ny_stdev must be a list or array of floats\n\n"
            raise ValueError(err)

        if self._num_obs is not None:
            if y_stdev.shape[0] != self._num_obs:
                err = "observation already loaded with a different size ({})\n".format(self._num_obs)
                raise ValueError(err)
        else:
            self._num_obs = y_stdev.shape[0]

        self._y_stdev = y_stdev

        self._fit_has_been_run = False


    @property
    def num_params(self):
        """
        Number of fit parameters.
        """

        if self._model_is_model_wrapper:
            return len(self._model.guesses)

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

        to_plot = s[:,np.array(keep_indexes,dtype=np.int)]
        #to_plot = np.swapaxes(to_plot,0,1)

        fig = corner.corner(to_plot,labels=names,range=corner_range,
                            truths=est_values,*args,**kwargs)

        return fig

    def write_samples(self,output_file):
        """
        Write the samples from the fit out to a pickle file.

        output_file: output pickle file to write to.
        """

        # See if the file exists already.
        if os.path.isfile(output_file):
            err = f"{output_file} exists.\n"
            raise FileExistsError(err)

        # If there are samples, write them out.
        if self.samples is not None:
            pickle.dump(self.samples,open(output_file,'wb'))

    def append_samples(self,sample_file=None,sample_array=None):
        """
        Append samples to the fit.  The new samples must be a float array
        with the shape: num_samples, num_parameters. This can come from a
        pickle file (sample_file) or array.  Only one of these can be specified.

        sample_file: Pickle file of the numpy array.
        sample_array: Array of samples.
        """

        # Nothing to do; no new samples specified
        if sample_file is None and sample_array is None:
            return

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
        has_err = False
        if isinstance(sample_array,np.ndarray) and sample_array.dtype == np.floating:
            if len(sample_array.shape) == 2:
                if self.num_params is not None:
                    if sample_array.shape[1] != self.num_params:
                        has_err = True
                else:
                    self._num_params = sample_array.shape[1]
            else:
                has_err = True
        else:
            has_err = True

        if has_err:
            err = "sample_array should be a float numpy array of shape\n"
            err += "(num_samples,num_param)\n"
            raise ValueError(err)

        if self.samples is None:
            err = "You can only append samples to  a fit that has already been done\.n"
            raise ValueError(err)

        warn = "\n\nThis function only checks to see if the input samples array\n"
        warn += "has the same number of parameters as the current number of\n"
        warn += "parameters. It does not check the identity of those parameters.\n"
        warn += "It will happily combine samples from one model with samples\n"
        warn += "from a different model.  It's up to you to make sure that does\n"
        warn += "not happen.  Happy Sampling. \U0001F600\n\n"
        warnings.warn(warn,UserWarning)

        # Concatenate the new samples to the existing samples
        self._samples = np.concatenate((self.samples,sample_array))

        self._update_estimates()
