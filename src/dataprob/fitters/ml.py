"""
Fitter subclass for performing maximum likelihood fits.
"""

from dataprob.fitters.base import Fitter
from dataprob.util.check import check_int

import numpy as np
import scipy.stats
import scipy.optimize as optimize

import warnings

class MLFitter(Fitter):
    """
    Fit the model to the data using nonlinear least squares.

    Standard deviation and ninety-five percent confidence intervals on parameter
    estimates are determined using the covariance matrix (Jacobian * residual
    variance) 
    """
    
    def fit(self,
            y_obs=None,
            y_std=None,
            num_samples=100000,
            **least_squares_kwargs):
        """
        Fit the model parameters to the data by maximum likelihood.

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
        num_samples : int
            number of samples for generating corner plot
        **least_squares_kwargs : 
            any remaining keyword arguments are passed as **kwargs to
            scipy.optimize.least_squares
        """
        
        self._num_samples = check_int(value=num_samples,
                                      variable_name="num_samples",
                                      minimum_allowed=0)

        super().fit(y_obs=y_obs,
                    y_std=y_std,
                    **least_squares_kwargs)                         

    def _fit(self,**kwargs):
        """
        Fit the parameters to the model.

        Parameters
        ----------
        kwargs : dict
            any keyword arguments are passed as **kwargs to
            scipy.optimize.least_squares
        """

        to_fit = self._model.unfixed_mask
        guesses = np.array(self._model.param_df.loc[to_fit,"guess"]).copy()
        bounds = np.array([self._model.param_df.loc[to_fit,"lower_bound"],
                           self._model.param_df.loc[to_fit,"upper_bound"]]).copy()
        # Do the actual fit
        def fn(*args): return -self._weighted_residuals(*args)
        self._fit_result = optimize.least_squares(fn,
                                                  x0=guesses,
                                                  bounds=bounds,
                                                  **kwargs)

        self._success = self._fit_result.success
        
        # Delete samples if they were present from a previous fit
        if hasattr(self,"_samples"):
            del self._samples
    
        self._update_fit_df()

    def _update_fit_df(self):
        """
        Recalculate the parameter estimates from any new samples.
        """
        
        estimate = self._fit_result.x

        # Extract standard error on the fit parameter from the covariance
        N = len(self._y_obs)
        P = len(self._fit_result.x)

        try:
            J = self._fit_result.jac
            cov = np.linalg.inv(2*np.dot(J.T,J))

            std = np.sqrt(np.diagonal(cov)) #variance

            # 95% confidence intervals from standard error
            z = scipy.stats.t(N-P-1).ppf(0.975)
            c1 = estimate - z*std
            c2 = estimate + z*std

            low_95 = []
            high_95 = []
            for i in range(P):
                low_95.append(c1[i])
                high_95.append(c2[i])

        except np.linalg.LinAlgError:
            w = "\n\nJacobian matrix was singular. Could not find parameter uncertainty.\n\n"
            warnings.warn(w)

            std = np.nan*np.ones(len(estimate),dtype=float)
            low_95 = np.nan*np.ones(len(estimate),dtype=float)
            high_95 = np.nan*np.ones(len(estimate),dtype=float)

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
    def samples(self):
        """
        Use the Jacobian spit out by least_squares to generate a whole bunch of
        fake samples.

        Approximate the covariance matrix as $(2*J^{T} \\dot J)^{-1}$, then perform
        cholesky factorization on the covariance matrix.  This can then be
        multiplied by random normal samples to create distributions that come
        from this covariance matrix.

        See:
        https://stackoverflow.com/questions/40187517/getting-covariance-matrix-of-fitted-parameters-from-scipy-optimize-least-squares
        https://stats.stackexchange.com/questions/120179/generating-data-with-a-given-sample-covariance-matrix
        """

        # If we already have samples, return them
        if hasattr(self,"_samples"):
            return self._samples

        # Return None if no fit has been run.        
        if not self._fit_has_been_run:
            return None
                
        try:
            J = self._fit_result.jac
            cov = np.linalg.inv(2*np.dot(J.T,J))
            chol_cov = np.linalg.cholesky(cov).T
        except np.linalg.LinAlgError:
            w = "\n\nJacobian matrix was singular. Could not generate parameter samples.\n\n"
            warnings.warn(w)

            # Return empty array
            return None

        unfixed = np.logical_not(np.array(self.fit_df["fixed"],dtype=bool))
        estimate = np.array(self.fit_df.loc[unfixed,"estimate"]).copy()
        self._samples = np.dot(np.random.normal(size=(self._num_samples,
                                                      chol_cov.shape[0])),
                                                chol_cov)
    
        self._samples = self._samples + estimate
        
        num_param = self._samples.shape[1]

        # above_mask is True for a given sample if all of the parameter values
        # are >= the lower bound for that sample
        lower_bound = np.array(self.fit_df.loc[unfixed,"lower_bound"],dtype=float)
        above_mask = np.sum(self._samples >= lower_bound,axis=1) == num_param

        # below_mak is True for a given sample if all of the parameter values
        # are <= the upper bound for that sample
        upper_bound = np.array(self.fit_df.loc[unfixed,"upper_bound"],dtype=float)
        below_mask = np.sum(self._samples <= upper_bound,axis=1) == num_param

        # Keep mask is True only if above_mask and below_mask are true for a 
        # given sample
        keep_mask = np.logical_and(above_mask,below_mask)

        # Get only samples that fit the condition 
        self._samples = self._samples[keep_mask,:]

        return self._samples


    def __repr__(self):
        """
        Output to show when object is printed or displayed in a jupyter 
        notebook.
        """

        out = ["MLFitter\n--------\n"]

        out.append(f"fit has been run: {self._fit_has_been_run}\n")
        if self._fit_has_been_run:
            out.append(f"fit results:\n")
            if self.success:
                for dataframe_line in repr(self.fit_df).split("\n"):
                    out.append(f"  {dataframe_line}")
                out.append("\n")
            else:
                out.append("  fit failed\n")

        return "\n".join(out)
