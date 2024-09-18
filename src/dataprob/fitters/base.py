"""
Fitter base class allowing different classes of fits.
"""

from dataprob.util.check import check_array
from dataprob.util.check import check_int

from dataprob.model_wrapper.model_wrapper import ModelWrapper
from dataprob.model_wrapper.wrap_function import wrap_function
from dataprob.util.read_spreadsheet import read_spreadsheet

from dataprob.util.get_fit_quality import get_fit_quality

import numpy as np
import pandas as pd

import pickle
import os
import copy

def _pretty_zeropad_str(N):
    """
    Make a string zero-pad based on the number of digits in a number.
    """

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
        Initialize the fitter.

        Parameters
        ----------
        some_function : callable
            A function that takes at least one argument and returns a float numpy
            array. Compare the outputs of this function against y_obs when doing
            the analysis. 
        fit_parameters : list, dict, str, pandas.DataFrame; optional
            fit_parameters lets the user specify information about the parameters 
            in the fit. 
        non_fit_kwargs : dict
            non_fit_kwargs are keyword arguments for some_function that should not
            be fit but need to be specified to non-default values. 
        vector_first_arg : bool, default=False
            If True, the first argument of the function is taken as a vector of 
            parameters to fit. All other arguments to some_function are treated as 
            non-fittable parameters. fit_parameters must then specify the names of
            each vector element. 
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

        # record y_obs if specified
        to_df = {}
        if y_obs is not None:
            to_df["y_obs"] = y_obs
        
        # record y_std if specified
        if y_std is not None:
            to_df["y_std"] = y_std

        # If both specified, turn into dataframe and store as data_df. Setter 
        # validates. 
        if len(to_df) == 2:
            self.data_df = pd.DataFrame(to_df)
        
        # If one specified, store in the existing data_df. data_df setter takes
        # care of validation
        elif len(to_df) == 1:

            data_df = self.data_df
            for column in to_df:
                data_df[column] = to_df[column]
            self.data_df = data_df

        # else here for completeness -- do not do anything if y_obs and y_std
        # are both zero. 
        else:
            pass

                
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
            y_std must either be specified here or in the data_df dataframe. 
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
        return -0.5*(np.sum((y_calc - self._y_obs)**2/sigma2 + np.log(2*np.pi*sigma2)))

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
        Observed y values for fit. 
        """

        try:
            return self._y_obs
        except AttributeError:
            return None


    @property
    def y_std(self):
        """
        Estimated standard deviation on observed y values. 
        """

        try:
            return self._y_std
        except AttributeError:
            return None

    @property
    def param_df(self):
        """
        Return a dataframe with fit parameters. 
        """

        return self._model.param_df
    
    @param_df.setter
    def param_df(self,param_df):
        self._model.param_df = param_df
    
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
        
            keep_mask = self._model.unfixed_mask

            estimate = np.array(self.fit_df.loc[keep_mask,"estimate"],
                                dtype=float).copy()
            out["y_calc"] = self.model(estimate)
            out["unweighted_residuals"] = self._unweighted_residuals(estimate)
            out["weighted_residuals"] = self._weighted_residuals(estimate)

        else:
                    
            # Try to generate a y_calc vector from the guesses. If this crashes,
            # it's okay; silently ignore and do not populate y_calc and 
            # residuals columns.
            try:        
                keep_mask = np.logical_not(self._model.param_df["fixed"])
                guesses = np.array(self._model.param_df.loc[keep_mask,"guess"],
                                   dtype=float).copy()
                
                out["y_calc"] = self.model(guesses)
                
                # Put these after y_calc. Even if y_calc succeeds, these could
                # fail if we don't have y_obs and y_std loaded. Build as much 
                # as we can before crashing. 
                out["unweighted_residuals"] = self._unweighted_residuals(guesses)
                out["weighted_residuals"] = self._weighted_residuals(guesses)

            except Exception as e:
                pass
        
        return pd.DataFrame(out)

    @data_df.setter
    def data_df(self,data_df):

        # Read dataframe
        data_df = read_spreadsheet(data_df)

        # make sure it has y_obs and y_std
        has_columns = np.sum(data_df.columns.isin(["y_obs","y_std"]))
        if has_columns != 2:
            err = "data_df must have both y_obs and y_std columns\n"
            raise ValueError(err)
        
        # Go through each column
        for c in ["y_obs","y_std"]:

            # start with the pandas caster to numeric as this is smart and
            # robust
            try:
                data_df[c] = pd.to_numeric(data_df[c])
            except Exception as e:
                err = f"Could not coerce all entries in the '{c}' column to float\n"
                raise ValueError(err) from e
            
            # then do a direct cast to float
            data_df[c] = data_df[c].astype(float)

            if np.sum(np.isnan(data_df[c])) > 0:
                err = "y_obs and y_std must not contain nan\n"
                raise ValueError(err)
            
            if np.sum(np.isinf(data_df[c])) > 0:
                err = "y_obs and y_std must be finite\n"
                raise ValueError(err)

        if np.sum(data_df["y_std"] <= 0) > 0:
            err = "all y_std values must be >= 0\n"
            raise ValueError(err)

        # Store y_obs and y_std
        self._y_obs = np.array(data_df["y_obs"],dtype=float)
        self._y_std = np.array(data_df["y_std"],dtype=float)

        # new y_obs, fit has not been run yet
        self._fit_has_been_run = False

    @property
    def non_fit_kwargs(self):
        """
        Return a dictionary with the keyword arguments to pass to the function
        that are not fit parameters.
        """
        
        return self._model.non_fit_kwargs

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
    def fit_quality(self):
        """
        """

        if not self.success:
            return None

        estimate = np.array(self.fit_df.loc[self._model.unfixed_mask,
                                            "estimate"],dtype=float).copy()

        out_df = get_fit_quality(residuals=self._weighted_residuals(estimate),
                                 num_param=estimate.shape[0],
                                 lnL=self.ln_like(estimate),
                                 success=self.success)

        return out_df



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
            estimate = np.array(self.fit_df["estimate"],dtype=float).copy()
            out["y_calc"] = self.model(estimate)
        else:
            estimate = np.array(self.fit_df["guess"],dtype=float).copy()
            out["y_calc"] = self.model(estimate)

        samples = self.samples
        if samples is not None:
            
            N = samples.shape[0]
            fmt_string = _pretty_zeropad_str(N)

            if num_samples >= N:
                i_values = np.arange(N)
            else:
                i_values = range(0,N,N//(num_samples-1))

            for i in i_values:
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