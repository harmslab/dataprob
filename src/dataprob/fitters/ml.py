"""
Fitter subclass for performing maximum likelihood fits.
"""

from .base import Fitter

import numpy as np
import scipy.stats
import scipy.optimize as optimize

import warnings

class MLFitter(Fitter):
    """
    Fit the model to the data using nonlinear least squares.

    Standard deviation and ninety-five percent confidence intervals on parameter
    estimates are determined using the covariance matrix (Jacobian * residual
    variance)  See:
    # http://stackoverflow.com/questions/14854339/in-scipy-how-and-why-does-curve-fit-calculate-the-covariance-of-the-parameter-es
    # http://stackoverflow.com/questions/14581358/getting-standard-errors-on-fitted-parameters-using-the-optimize-leastsq-method-i
    """
    def __init__(self,num_samples=100000):
        """
        Initialize the fitter.

        Parameters
        ----------
        num_samples : int
            number of samples for generating corner plot
        """

        super().__init__()

        self._fit_type = "maximum likelihood"
        self._num_samples = num_samples

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
        guesses = np.array(self._model.param_df.loc[to_fit,"guess"])
        bounds = np.array([self._model.param_df.loc[to_fit,"lower_bound"],
                           self._model.param_df.loc[to_fit,"upper_bound"]])

        # Do the actual fit
        fn = lambda *args: -self.weighted_residuals(*args)
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

            stdev = np.sqrt(np.diagonal(cov)) #variance

            # 95% confidence intervals from standard error
            z = scipy.stats.t(N-P-1).ppf(0.975)
            c1 = estimate - z*stdev
            c2 = estimate + z*stdev

            low_95 = []
            high_95 = []
            for i in range(P):
                low_95.append(c1[i])
                high_95.append(c2[i])

        except np.linalg.LinAlgError:
            warning = "\n\nJacobian matrix was singular.\n"
            warning += "Could not estimate parameter uncertainty.\n"
            warning += "Consider using the Bayesian sampler.\n"
            warnings.warn(warning)

            stdev = np.nan*np.ones(len(estimate),dtype=float)
            low_95 = np.nan*np.ones(len(estimate),dtype=float)
            high_95 = np.nan*np.ones(len(estimate),dtype=float)
        
        self._fit_df["estimate"] = estimate
        self._fit_df["std"] = stdev
        self._fit_df["low_95"] = low_95
        self._fit_df["high_95"] = high_95


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
            warning = "\n\nJacobian matrix was singular.\n"
            warning += "Could not estimate parameter uncertainty.\n"
            warning += "Consider using the Bayesian sampler.\n"
            warnings.warn(warning)

            # Return empty array
            return None

        estimate = np.array(self.fit_df["estimate"])
        self._samples = np.dot(np.random.normal(size=(self._num_samples,
                                                        chol_cov.shape[0])),
                                                chol_cov)
        

        self._samples = self._samples + estimate

        return self._samples


    def __repr__(self):

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
