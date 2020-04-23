__description__ = \
"""
Fitter subclass for performing maximum likelihood fits.
"""
__author__ = "Michael J. Harms"
__date__ = "2017-05-10"

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

        num_samples: number of samples for generating corner plot
        """

        Fitter.__init__(self)

        self.fit_type = "maximum likelihood"
        self._num_samples = num_samples

    def fit(self,model=None,guesses=None,y_obs=None,bounds=None,names=None,y_stdev=None,**kwargs):
        """
        Fit the parameters.

        Parameters
        ----------

        model : callable
            model to fit.  model should take "guesses" as its only argument.
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
            is assigned an error of 1
        **kwargs : any remaining keywaord arguments are passed as **kwargs to
            scipy.optimize.least_squares
        """

        self._preprocess_fit(model,guesses,y_obs,bounds,names,y_stdev)

        # Do the actual fit
        fn = lambda *args: -self.weighted_residuals(*args)
        self._fit_result = optimize.least_squares(fn,
                                                  x0=self._guesses,
                                                  bounds=self._bounds,
                                                  **kwargs)
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

            self._ninetyfive = []
            for i in range(P):
                self._ninetyfive.append([c1[i],c2[i]])
            self._ninetyfive = np.array(self._ninetyfive)

        except np.linalg.LinAlgError:
            warning = "\n\nJacobian matrix was singular.\n"
            warning += "Could not estimate parameter uncertainty.\n"
            warning += "Consider using the Bayesian sampler.\n"
            warnings.warn(warning)

            self._stdev = np.nan*np.ones(len(self._estimate),dtype=np.float)
            self._ninety_five = np.nan*np.ones((len(self._estimate),2),dtype=np.float)

        self._success = self._fit_result.success

    @property
    def fit_info(self):
        """
        Return information about the fit.
        """

        return {}

    @property
    def samples(self):
        """
        Use the Jacobian spit out by least_squares to generate a whole bunch of
        fake samples.

        Approximate the covariance matrix as $(2*J^{T} \dot J)^{-1}$, then perform
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
                J = self._fit_result.jac

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
