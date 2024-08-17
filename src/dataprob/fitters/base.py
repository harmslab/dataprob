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
import copy

from dataprob.check import check_array
from dataprob.check import check_float
from dataprob.check import check_int

from dataprob.model_wrapper.model_wrapper import ModelWrapper
from dataprob.model_wrapper.wrap_function import wrap_function

def _pretty_zeropad_str(N):

    num_digits = len(f"{N}") + 1
    fmt_string = "s{:0" + f"{num_digits}" + "d}"
    return fmt_string

class Fitter:
    """
    Base class for fits/analyses using a dataprob function.
    """

    def __init__(self,
                 some_function,
                 fit_parameters=None,
                 non_fit_kwargs=None,
                 vector_first_arg=False):
        """
        Init function for the class.
        """

        # Load the model. Copy in ModelWrapper if passed in; otherwise, create
        # from arguments. 
        if issubclass(type(some_function),ModelWrapper):
            self._model = copy.deepcopy(some_function)
        else:        
            self._model = wrap_function(some_function=some_function,
                                        fit_parameters=fit_parameters,
                                        non_fit_kwargs=non_fit_kwargs,
                                        vector_first_arg=vector_first_arg)
        
        # Initialize the fit df now that we have a model
        self._initialize_fit_df()

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
            y_obs=None,
            y_std=None,
            **kwargs):
        """
        Fit the parameters.

        Parameters
        ----------
        y_obs : numpy.ndarray
            observations in a numpy array of floats that matches the shape
            of the output of some_function set when initializing the fitter. 
            nan values are not allowed. y_obs must either be specified here 
            or in the data_df dataframe. 
        y_std : numpy.ndarray
            standard deviation of each observation. nan values are not allowed.
            If not specified, all points are assigned an uncertainty of
            0.1*mean(y_obs). 
        **kwargs : any remaining keyword arguments are passed as **kwargs to
            the core engine (optimize.least_squares or emcee.EnsembleSampler)
        """

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

        y_calc = self._model.fast_model(param)
        return y_calc - self._y_obs

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

        y_calc = self._model.fast_model(param)
        return (y_calc - self._y_obs)/self._y_std

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

        y_calc = self._model.fast_model(param)
        sigma2 = self._y_std**2
        return -0.5*(np.sum((y_calc - self._y_obs)**2/sigma2 + np.log(sigma2)))

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
        Model to use for calculating y_calc given parameters. 
        """

        return self._model.model

             
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

        return self._model.param_df
        
        
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
        df["guess"] = self.param_df["guess"]
        df["fixed"] = self.param_df["fixed"]
        df["lower_bound"] = self.param_df["lower_bound"]
        df["upper_bound"] = self.param_df["upper_bound"]
        df["prior_mean"] = self.param_df["prior_mean"]
        df["prior_std"] = self.param_df["prior_std"]

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

        return self._fit_df
        

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

        # Check sanity of sample_array;
        try:
            sample_array = np.array(sample_array,dtype=float)
        except Exception as e:
            err = "sample_array should be a float numpy array\n"
            raise ValueError(err) from e
        
        if len(sample_array.shape) != 2:
            err = "sample_array should have dimensions (num_samples,num_params)\n"
            raise ValueError(err)

        if sample_array.shape[1] != self.num_params:
            err = "sample_array should have dimensions (num_samples,num_params)\n"
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
            out["y_calc"] = self.model(np.array(self.fit_df["estimate"],
                                                dtype=float))

        samples = self.samples
        if samples is not None:
            
            N = samples.shape[0]
            fmt_string = _pretty_zeropad_str(N)

            for i in range(0,N,N//(num_samples-1)):
                key = fmt_string.format(i)
                out[key] = self.model(self.samples[i])
    
        return pd.DataFrame(out)

    @property
    def num_params(self):
        """
        Number of fit parameters. If model has not been defined, will be None. 
        """

        self._model.finalize_params()
        return np.sum(self._model.unfixed_mask)

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