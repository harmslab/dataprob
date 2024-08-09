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

        # Do the actual fit
        fn = lambda *args: -self.weighted_residuals(*args)
        self._fit_result = optimize.least_squares(fn,
                                                  x0=self.guesses,
                                                  bounds=self.bounds,
                                                  **kwargs)

        self._success = self._fit_result.success

        self._update_estimates()

    def _update_estimates(self):
        """
        Recalculate the parameter estimates from any new samples.
        """
        
        self._estimate = self._fit_result.x

        # Extract standard error on the fit parameter from the covariance
        N = len(self._y_obs)
        P = len(self._fit_result.x)

        try:
            J = self._fit_result.jac
            cov = np.linalg.inv(2*np.dot(J.T,J))

            self._stdev = np.sqrt(np.diagonal(cov)) #variance)

            # 95% confidence intervals from standard error
            z = scipy.stats.t(N-P-1).ppf(0.975)
            c1 = self._estimate - z*self._stdev
            c2 = self._estimate + z*self._stdev

            self._ninetyfive = [[],[]]
            for i in range(P):
                self._ninetyfive[0].append(c1[i])
                self._ninetyfive[1].append(c2[i])
            self._ninetyfive = np.array(self._ninetyfive)

        except np.linalg.LinAlgError:
            warning = "\n\nJacobian matrix was singular.\n"
            warning += "Could not estimate parameter uncertainty.\n"
            warning += "Consider using the Bayesian sampler.\n"
            warnings.warn(warning)

            self._stdev = np.nan*np.ones(len(self._estimate),dtype=float)
            self._ninety_five = np.nan*np.ones((2,len(self._estimate)),dtype=float)


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

        try:
            return self._samples
        except AttributeError:

            try:
                # Return None if no fit has been run.
                try:
                    J = self._fit_result.jac
                except AttributeError:
                    return None

                cov = np.linalg.inv(2*np.dot(J.T,J))
                chol_cov = np.linalg.cholesky(cov).T
            except np.linalg.LinAlgError:
                warning = "\n\nJacobian matrix was singular.\n"
                warning += "Could not estimate parameter uncertainty.\n"
                warning += "Consider using the Bayesian sampler.\n"
                warnings.warn(warning)

                # Return empty array
                return np.array([])

            self._samples = np.dot(np.random.normal(size=(self._num_samples,
                                                          chol_cov.shape[0])),
                                                    chol_cov)
            self._samples = self._samples + self.estimate

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
