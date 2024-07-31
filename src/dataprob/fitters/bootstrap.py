"""
Fitter subclass for performing bootstrap analyses.
"""

from .base import Fitter

import numpy as np
import scipy.optimize

import sys

class BootstrapFitter(Fitter):
    """
    Perform the fit many times, sampling from uncertainty in each measurement.
    """

    def __init__(self,num_bootstrap=100,perturb_size=1.0,exp_err=False,verbose=False):
        """
        Perform the fit many times, sampling from uncertainty in each measured
        heat.

        Parameters
        ----------
        num_bootstrap : int
            Number of bootstrap samples to do
        perturb_size : float
            Standard deviation of random samples for heats.  Ignored if exp_err
            is specified.
        exp_err : bool
            Use experimental estimates of heat uncertainty.  If specified, overrides
            perturb_size.
        verbose : bool
            Give verbose output.
        """

        super(BootstrapFitter,self).__init__()

        self._num_bootstrap = num_bootstrap
        self._perturb_size = perturb_size
        self._exp_err = exp_err
        self._verbose = verbose

        self.fit_type = "bootstrap"

    def _fit(self,**kwargs):
        """
        Fit the parameters.

        Parameters
        ----------
        kwargs : dict
            any keyword arguments are passed as **kwargs to 
            scipy.optimize.least_squares
        """

        # Create array to store bootstrap replicates
        samples = np.zeros((self._num_bootstrap,len(self.guesses)),dtype=float)

        original_y_obs = np.copy(self._y_obs)

        # Go through bootstrap reps
        for i in range(self._num_bootstrap):

            if self._verbose and i != 0 and i % 100 == 0:
                print("Bootstrap {} of {}".format(i,self._num_bootstrap))
                sys.stdout.flush()

            # Add random error to each sample
            self.y_obs = original_y_obs + np.random.normal(0.0,self.y_stdev)

            # Do the fit
            fit = scipy.optimize.least_squares(self.unweighted_residuals,
                                               x0=self.guesses,
                                               bounds=self.bounds,
                                               **kwargs)

            # record the fit results
            samples[i,:] = fit.x

        self._y_obs = np.copy(original_y_obs)

        if self.samples is None:
            self._samples = samples

        # If samples have already been done, append to them.
        else:
            self._samples = np.concatenate((self._samples,samples))

        self._fit_result = self._samples

        self._update_estimates()

    def _update_estimates(self):
        """
        Recalculate the parameter estimates from the new samples.
        """

        # mean of bootstrap samples
        self._estimate = np.mean(self._samples,axis=0)

        # standard deviation from bootstrap samples
        self._stdev = np.std(self._samples,axis=0)

        # 95% from bootstrap samples
        self._ninetyfive = [[],[]]
        for i in range(self._samples.shape[1]):
            lower = np.percentile(self._samples[:,i], 2.5)
            upper = np.percentile(self._samples[:,i],97.5)
            self._ninetyfive[0].append(lower)
            self._ninetyfive[1].append(upper)
        self._ninetyfive = np.array(self._ninetyfive)

        self._success = True

    @property
    def fit_info(self):
        """
        Return information about the fit.
        """

        output = {}

        output["Num bootstrap"] = self._num_bootstrap
        output["Perturb size"] = self._perturb_size
        output["Use experimental error"] = self._exp_err

        return output
