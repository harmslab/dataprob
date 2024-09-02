"""
Fitter subclass for performing bootstrap analyses.
"""

from dataprob.fitters.base import Fitter
from dataprob.util.check import check_int
from dataprob.util.stats import get_kde_max

import numpy as np
import scipy
from tqdm.auto import tqdm

import warnings

class BootstrapFitter(Fitter):
    """
    Perform the fit many times, sampling from uncertainty in each measurement.
    """

    def fit(self,
            y_obs=None,
            y_std=None,
            num_bootstrap=100,
            **least_squares_kwargs):
        """
        Fit the model parameters to the data by maximum likelihood, sampling 
        uncertainty in observation values by bootstrap. 

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
        num_bootstrap : int
            Number of bootstrap samples to run
        **least_squares_kwargs : 
            any remaining keyword arguments are passed as **kwargs to
            scipy.optimize.least_squares
        """
        
        self._num_bootstrap = check_int(value=num_bootstrap,
                                        variable_name="num_bootstrap",
                                        minimum_allowed=2)

        super().fit(y_obs=y_obs,
                    y_std=y_std,
                    **least_squares_kwargs)    

    def _fit(self,**kwargs):
        """
        Fit the parameters.

        Parameters
        ----------
        kwargs : dict
            any keyword arguments are passed as **kwargs to 
            scipy.optimize.least_squares
        """

        # Grab un-fixed guesses and bounds
        to_fit = self._model.unfixed_mask
        guesses = np.array(self._model.param_df.loc[to_fit,"guess"]).copy()
        bounds = np.array([self._model.param_df.loc[to_fit,"lower_bound"],
                           self._model.param_df.loc[to_fit,"upper_bound"]]).copy()

        # Create array to store bootstrap replicates
        samples = np.zeros((self._num_bootstrap,len(guesses)),dtype=float)

        # Record y_obs
        original_y_obs = np.copy(self._y_obs)

        # Define function to regress against
        def fn(*args): return -self._unweighted_residuals(*args)
        
        # Go through bootstrap reps
        problems = []
        for i in tqdm(range(self._num_bootstrap)):
        
            # Create updated version of y_obs sampled from y_std
            self._y_obs = original_y_obs + np.random.normal(0.0,self._y_std)

            # Do regression
            try:
                fit = scipy.optimize.least_squares(fn,
                                                   x0=guesses,
                                                   bounds=bounds,
                                                   **kwargs)
            except Exception as e:
                problems.append(str(e))
                samples[i,:] = np.nan
                continue

            # Record the fit results. If the fit fails, record np.nan
            if fit.success:
                samples[i,:] = fit.x
            else:
                samples[i,:] = np.nan

        # Restore y_obs from our stored copy
        self._y_obs = np.copy(original_y_obs)

        # If no samples yet, store them. Otherwise, append them to the existing
        # samples. 
        if self.samples is None:
            self._samples = samples
        else:
            self.append_samples(sample_array=samples)

        # Record the current stats on the number of samples and number that 
        # failed and place in _fit_result. 
        total_samples = self.samples.shape[0]
        num_failed = np.sum(np.isnan(self.samples[:,0]))
        num_success = total_samples - num_failed
        self._fit_result = {"total_samples":total_samples,
                            "num_success":num_success,
                            "num_failed":num_failed}

        # warn if a fit failed to converge
        if num_failed > 0:
            w = f"\n\nOnly {num_success} of {total_samples} fits were successful.\n\n"
            
            if len(problems) > 0:
                prob_types, prob_counts = np.unique(problems,
                                                    return_counts=True)
                w += "The fitter threw the following errors:\n"
                for i in range(len(prob_types)):
                    w += f"  '{prob_types[i]}' {prob_counts[i]} times.\n"
                
            warnings.warn(w)

        # If at least two replicates worked, record this as success
        if num_success > 2:
            self._success = True
        else:
            self._success = False

        if self._success:
            self._update_fit_df()

    def _update_fit_df(self):
        """
        Recalculate the parameter estimates from any new samples.
        """
        
        samples = self.samples
        good_mask = np.sum(np.isnan(samples),axis=1) == 0
        samples = samples[good_mask,:]
        if samples.shape[0] < self.samples.shape[1] + 1:
            err = f"_update_fit_df requires at least num_param + 1 non-nan samples. The\n"
            err += f".samples array has {samples.shape[0]} non-nan parameters.\n"
            raise ValueError(err)

        # Get mean and standard deviation
        estimate = get_kde_max(self._samples)
        std = np.std(samples,axis=0)

        # Calculate 95% confidence intervals
        lower = int(round(0.025*samples.shape[0],0))
        upper = int(round(0.975*samples.shape[0],0))

        # For samples less than ~100, the rounding above will make the
        # the upper cutoff the number of samples, and thus lead to an index 
        # error below. 
        if upper >= samples.shape[0]:
            upper = samples.shape[0] - 1

        low_95 = []
        high_95 = []
        for i in range(samples.shape[1]):
            sorted_samples = np.sort(samples[:,i])
            low_95.append(sorted_samples[lower])
            high_95.append(sorted_samples[upper])

        # Get finalized parameters from param_df in case they were updated 
        # after the model was set and the fit_df created. 
        for col in ["guess","fixed","lower_bound","upper_bound","prior_mean",
                    "prior_std"]:
            self._fit_df[col] = self.param_df[col]

        fixed = np.array(self._fit_df["fixed"],dtype=bool).copy()
        unfixed = np.logical_not(fixed)

        self._fit_df.loc[unfixed,"estimate"] = estimate
        self._fit_df.loc[fixed,"estimate"] = self._fit_df.loc[fixed,"guess"]
        self._fit_df.loc[unfixed,"std"] = std
        self._fit_df.loc[unfixed,"low_95"] = low_95
        self._fit_df.loc[unfixed,"high_95"] = high_95

    
    @property
    def fit_info(self):
        """
        Return information about the fit.
        """

        output = {}

        if hasattr(self,"_num_bootstrap"):
            output["Num bootstrap"] = self._num_bootstrap

        return output
    
    def __repr__(self):
        
        out = ["BootstrapFitter\n---------------\n"]

        out.append("Fit parameters:\n")
        for k in self.fit_info:
            out.append(f"  {k}: {self.fit_info[k]}")

        out.append(f"\nfit has been run: {self._fit_has_been_run}\n")

        if self._fit_has_been_run:
            out.append(f"fit results:\n")
            if self.success:
                for dataframe_line in repr(self.fit_df).split("\n"):
                    out.append(f"  {dataframe_line}")
                out.append("\n")
            else:
                out.append("  fit failed\n")

        return "\n".join(out)
