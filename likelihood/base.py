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

import re, inspect

import likelihood

class LikelihoodError(Exception):
    pass

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
        self._model_is_model_wrapper = False

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
                raise LikelihoodError(err)


    def _preprocess_fit(self,model,guesses,y_obs,bounds=None,names=None,y_stdev=None):
        """
        Preprocess the user inputs to self.fit(), checking for sanity and
        putting in default values if none specificed by user.
        """

        # Record model, check for preloaded model, or fail.
        if model is not None:
            self.model = model
        else:
            if self.model is None:
                err = "model must be specified before fit\n"
                raise LikelihoodError(err)

        # Record guesses, grab from ModelWrapper model, or fail.
        if guesses is not None:
            self.guesses = guesses
        else:
            if self.guesses is None:
                if self._model_is_model_wrapper:
                    self.guesses = self.model.__self__.guesses
                else:
                    err = "parameter guesses must be specified before fit\n"
                    raise LikelihoodError(err)

        # Record bounds, grab from ModelWrapper model, or make infinite
        if bounds is not None:
            self.bounds = bounds
        else:
            if self.bounds is None:
                if self._model_is_model_wrapper:
                    self.bounds = self.model.__self__.bounds
                else:
                    tmp = np.ones(len(self.guesses))
                    self.bounds = [-np.inf*tmp,np.inf*tmp]

        # Record names, grab from ModelWrapper model, or use default
        if names is not None:
            self.names = names
        else:
            if self.names is None:
                if self._model_is_model_wrapper:
                    self.names = self.model.__self__.names
                else:
                    self.names = ["p{}".format(i) for i in range(len(self.guesses))]

        # Record y_obs, check for preloaded, or fail
        if y_obs is not None:
            self.y_obs = y_obs
        else:
            if self.y_obs is None:
                err = "y_obs must be specified before fit\n"
                raise LikelihoodError(err)

        # Record y_stdev, check for preloaded, or use default
        if y_stdev is not None:
            self.y_stdev = y_stdev
        else:
            if self.y_stdev is None:
                self.y_stdev = np.ones(len(self.y_obs),dtype=np.float)

        # No fit has been run
        self._success = None

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

        self._preprocess_fit(model,guesses,y_obs,bounds,names,y_stdev)
        self._sanity_check("fit can be done",["model","y_obs","y_stdev"])

        self._fit(**kwargs)

        self._post_fit()

    def _fit(self,**kwargs):

        pass

    def _post_fit(self):

        # Load the fit results
        if self._model_is_model_wrapper:
            self.model.__self__.load_fit_result(self)

    @property
    def fit_as_df(self):

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

            m = self.model.__self__

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
                    out_dict["low_95"].append(m.fit_parameters[p].ninetyfive[0])
                    out_dict["high_95"].append(m.fit_parameters[p].ninetyfive[1])
                    out_dict["guess"].append(m.fit_parameters[p].guess)
                    out_dict["lower_bound"].append(m.fit_parameters[p].bounds[0])
                    out_dict["upper_bound"].append(m.fit_parameters[p].bounds[1])

        else:

            for i in range(len(self.names)):
                out_dict["param"].append(self.names[i])
                out_dict["estimate"].append(self.estimate[i])
                out_dict["stdev"].append(self.stdev[i])
                out_dict["low_95"].append(self.ninetyfive[i,0])
                out_dict["high_95"].append(self.ninetyfive[i,1])
                out_dict["guess"].append(self.guesses[i])
                out_dict["lower_bound"].append(self.bounds[0,i])
                out_dict["upper_bound"].append(self.bounds[1,i])

        return pd.DataFrame(out_dict)


    def unweighted_residuals(self,param):
        """
        Calculate residuals.
        """

        self._sanity_check("residuals can be calculated",["model","y_obs"])

        y_calc = self.model(param)
        return self._y_obs - y_calc


    def weighted_residuals(self,param):
        """
        Calculate weighted residuals.
        """

        self._sanity_check("residuals can be calculated",["model","y_obs","y_stdev"])

        y_calc = self.model(param)
        return (self.y_obs - y_calc)/self.y_stdev

    def ln_like(self,param):
        """
        Log likelihood of function given parameters.
        """

        self._sanity_check("likelihood can be calculated",["model","y_obs","y_stdev"])

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
            return self._model
        except AttributeError:
            return None

    @model.setter
    def model(self,model):
        """
        Setter for "model" attribute.
        """

        # If this is a ModelWrapper instance, grab the model method rather than
        # the model instance
        if isinstance(model,likelihood.model_wrapper.ModelWrapper):
            model = model.model

        # If the model is a method of a ModelWrapper instance, record this so
        # the Fitter object knows it can get guesses, bounds, and names from
        # the ModelWrapper if necessary.
        self._model_is_model_wrapper = False
        try:
            if model.__qualname__.startswith("ModelWrapper"):
                self._model_is_model_wrapper = True
        except AttributeError:
            pass

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

        self._model = model


    @property
    def guesses(self):
        """
        Guesses for fit parameters.
        """

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

    @property
    def bounds(self):
        """
        Bounds for fit parameters.
        """

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

    @property
    def names(self):
        """
        Parameter names for fit parameters.
        """

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

            # Will through a type error if the user puts in a single value
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


    @property
    def num_params(self):
        """
        Number of fit parameters.
        """

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
        if s is None:
            err = "\n\nFit did not produce samples for generation of a corner plot.\nCheck warnings.\n"
            raise RuntimeError(err)

        keep_indexes = []
        corner_range = []
        names = []
        est_values = []
        for i in range(s.shape[1]):

            # look for patterns to skip
            if skip_pattern.search(self._names[i]):
                print("not doing corner plot for parameter ",self._names[i])
                continue

            names.append(self._names[i])
            keep_indexes.append(i)
            corner_range.append(tuple([np.min(s[:,i])-0.5,np.max(s[:,i])+0.5]))

            est_values.append(self.estimate[i])

        to_plot = s[:,np.array(keep_indexes,dtype=np.int)]
        #to_plot = np.swapaxes(to_plot,0,1)

        fig = corner.corner(to_plot,labels=names,range=corner_range,
                            truths=est_values,*args,**kwargs)

        return fig
