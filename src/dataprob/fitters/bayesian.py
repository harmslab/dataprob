"""
Fitter subclass for performing bayesian (MCMC) parameter estimation.
"""

from dataprob.fitters.base import Fitter
from dataprob.check import check_int
from dataprob.check import check_float
from dataprob.check import check_bool
from dataprob.check import check_array

import emcee

import numpy as np
import scipy.optimize as optimize
from scipy import stats

import multiprocessing
import warnings

def _find_normalization(scale,rv,**kwargs):
    """
    This method finds an offset to add to the output of rv.logpdf that 
    makes the total area under the pdf curve 1.0. 

    Parameters
    ----------
    scale : float
        scale argument to scipy.stats.rv
    rv : rv_continuous
        object for calculating logpdf
    kwargs : dict
        kwargs are passed to rv_continuous to initialize

    Returns
    -------
    offset : float
        offset to add to frozen_rv.logpdf(x) to normalize pdf to 1.0 

    Notes
    -----

    Calculate the minimum difference between floats we can represent using our
    data type. This sets the bin width for a discrete approximation of the
    continuous distribution.

        res = np.finfo(dtype).resolution
    
    Calculate the sum of the un-normalized pdf over a range spanning zero. We
    use a range of -num_to_same*res -> num_to_sample*res, taking steps of res:

        frozen_rv = rv(loc=0,scale=scale,**kwargs)
        x = np.arange(-num_to_sample*res,
                      num_to_sample*(res + 1),
                      res)
        un_norm_prob = np.sum(frozen_rv.pdf(loc=x))

    Calculate the difference in the cdf between num_to_sample*res and 
    -num_to_sample*res. This is the normalized probability of the slice
    centered on zero we calculated above. 

        norm_prob = cdf(num_to_sample*res) - cdf(-num_to_sample*res). 

    The ratio of norm_prob and un_norm_prob is a scalar that converts from raw
    pdf calculations to normalized probabilities. 

        norm_prob_x = frozen_rv.pdf(x)*norm_prob/un_norm_prob

    Since we care about log versions of this, it becomes:

        log_norm_prob_x = frozen_rv.logpdf(x) + log(norm_prob) - log(un_norm_prob)

    We can define a pre-calculated offset;

        offset = log(norm_prob) - log(un_norm_prob) 

    So, finally:

        log_norm_prob_x = frozen_rv.logpdf(x) + offset
    """

    # Create frozen distribution located at 0.0
    centered_rv = rv(loc=0,scale=scale,**kwargs)

    # Get smallest float step (resolution) for this datatype on this
    # platform.
    res = np.finfo(centered_rv.cdf(0).dtype).resolution

    num_to_sample = int(np.round(1/res/1e9,0))

    # Calculate prob of -1000res -> 1000res using cdf
    cdf = centered_rv.cdf(num_to_sample*res) - centered_rv.cdf(-num_to_sample*res)

    # Calculate pdf over this interval with a step size of res
    pdf = np.sum(centered_rv.pdf(np.linspace(-num_to_sample*res,
                                             num_to_sample*res,
                                             2*num_to_sample + 1)))
    
    # This normalizes logpdf. It's the log version of:
    # norm_prob_x = rv.pdf(x)*cdf/pdf. 
    offset = np.log(cdf) - np.log(pdf)
    
    return offset

def _reconcile_bounds_and_priors(bounds,frozen_rv):
    """
    Figure out how much bounds trim off priors and return amount to add to the 
    prior offset area to set to pdf area to 1.

    Parameters
    ----------
    bounds : list-like or None
        bounds applied to parameter
    
    Returns
    -------
    offset : float
        offset to add to np.logpdf(x) that accounts for the fact that bounds 
        may have trimmed some of the probability density and normalizes the 
        area of the pdf to 1.0. 
    """

    # bounds specified, no offset to area
    if bounds is None:
        return 0.0

    # Parse bounds list.
    left = float(bounds[0])
    right = float(bounds[1])
    
    # left and right the same. prob of this value is 1 (np.log(1) = 0)
    if left == right:
        w = f"left and right bounds {bounds} are identical\n"
        warnings.warn(w)
        return 0.0
    
    # Calculate the amount the bounds trim off the top and the bottom of the 
    # distribution. 
    left_trim = frozen_rv.cdf(left) 
    right_trim = frozen_rv.sf(right)

    # Figure out how much we need to scale the area up given we lost some
    # of the tails. 
    remaining = 1 - (left_trim + right_trim)

    # If remaining ends up zero, the left and right edges of the bounds
    # are identical within numerical error. prob of this value is 1 (np.log(1) = 0)
    if remaining == 0:
        w = f"left and right bounds {bounds} are numerically identical\n"
        warnings.warn(w)
        return 0.0

    # Return amount to scale the area by
    return np.log(1/remaining)

def _find_uniform_value(bounds):
    """
    Find the log probability for a specific value between bounds to use in a 
    uniform prior. 

    Parameters
    ----------
    bounds : list-like, optional
        list like float of two values. function assumes these are non-nanf 
        floats where bounds[1] >= bounds[0]. If bounds is None, assume 
        infinite bounds. 
    
    The idea here is to find the maximum finite width the parameter can occupy
    (bound[1] - bound[0]) and then to divide that by the number of steps based 
    on our numerical resolution. If our resolution was 0.1 and we were between 
    0 and 1, there would be 11 values, so the uniform prior would be ln(1/11).
    The log prior ends up being: np.log(resolution/width)

    For finite bounds, this is:
    
        np.log(resolution) - np.log(upper - lower)

    For infinite bounds, this is:
        
        np.log(resolution) - (np.log(scalar) + log(max_positive_finite_float)).

    max_positive_finite_float is the largest number we can represent. scalar
    ranges (0,2]. It would be 2 for the bounds (-infinity,infinity), because 
    this covers a span 2*max_positive_finite_float. To avoid overflow, we 
    represent as logs (ln(2*max) --> ln(2) + ln(max)). Scalar values lower than
    2 represent smaller chunks of the finite number line. A value of 1 would 
    be half the number line (-infinity,0), (0,infinity). The value approaches
    zero for the bounds (-infinity, -infinity + resolution) and 
    (infinity - resolution,infinity). 
    """

    # float architecture information
    finfo = np.finfo(np.ones(1,dtype=float)[0].dtype)
    log_resolution = np.log(finfo.resolution)
    log_max_value = np.log(finfo.max)
    max_value = finfo.max

    # width is 2*max_value --> log(2) + log_max_value
    if bounds is None:
        return log_resolution - (np.log(2) + log_max_value)

    # Parse bounds list
    left = float(bounds[0])
    right = float(bounds[1])

    # left and right the same, probability is 1.0 (log(P) = 0) for this value
    if left == right:
        w = f"left and right bounds {bounds} are identical\n"
        warnings.warn(w)
        return 0.0
    
    # width is 2*max_value --> log(2) + log_max_value
    if np.isinf(left) and np.isinf(right):
        return log_resolution - (np.log(2) + log_max_value)

    # Figure out scalars (see docstring)
    if np.isinf(left):

        # exactly half the number line; scalar = 1
        if right == 0:
            return log_resolution - log_max_value
        
        # scalar is less than 1 if left and right are both to the left of zero,
        # more than 1 if right is above zero
        if right < 0:
            scalar = 1 - np.abs(right/max_value)
        else:
            scalar = 1 + np.abs(right/max_value)
        
        return log_resolution - (np.log(scalar) + log_max_value)
    
    if np.isinf(right):
        
        # exactly half the number line; scalar = 1
        if left == 0:
            return log_resolution - log_max_value
        
        # scalar is less than 1 if left and right are both to the right of zero,
        # more than 1 if left is below zero
        if left > 0:
            scalar = 1 - np.abs(left/max_value)
        else:
            scalar = 1 + np.abs(left/max_value)
        
        return log_resolution - (np.log(scalar) + log_max_value)

    # simple case with finite bounds. resolution/bound_width
    return np.log(finfo.resolution) - np.log((right - left))


class BayesianSampler(Fitter):
    """
    Use Bayesian MCMC to sample parameter space. 
    """
    def __init__(self,
                 num_walkers=100,
                 initial_walker_spread=1e-4,
                 ml_guess=True,
                 num_steps=100,
                 burn_in=0.1,
                 num_threads=1):
        """
        Initialize the bayesian sampler.

        Parameters
        ----------
        num_walkers : int > 0
            how many markov chains to have in the analysis
        initial_walker_spread : float
            each walker is initialized with parameters sampled from normal
            distributions with mean equal to the initial guess and a standard
            deviation of guess*initial_walker_spread
        ml_guess : bool
            if true, do an ML optimization to get the initial guess
        num_steps:
            number of steps to run the markov chains
        burn_in : float between 0 and 1
            fraction of samples to discard from the start of the run
        num_threads : int
            number of threads to use.  if `0`, use the total number of cpus. 
            [NOT YET IMPLEMENTED]
        """

        super().__init__()

        # Set keywords, validating as we go
        self._num_walkers = check_int(value=num_walkers,
                                      variable_name="num_walkers",
                                      minimum_allowed=1)
        self._initial_walker_spread = check_float(value=initial_walker_spread,
                                                  variable_name="initial_walker_spread",
                                                  minimum_allowed=0,
                                                  minimum_inclusive=False)               
        self._ml_guess = check_bool(value=ml_guess,
                                    variable_name="ml_guess")
        self._num_steps = check_int(value=num_steps,
                                    variable_name="num_steps",
                                    minimum_allowed=1)
        self._burn_in = check_float(value=burn_in,
                                    variable_name="burn_in",
                                    minimum_allowed=0,
                                    maximum_allowed=1,
                                    minimum_inclusive=False,
                                    maximum_inclusive=False)

        # Deal with number of threads
        num_threads = check_int(value=num_threads,
                                variable_name="num_threads",
                                minimum_allowed=0)                
        if num_threads == 0:
            num_threads = multiprocessing.cpu_count()

        if num_threads != 1:
            err = "multithreading has not yet been implemented (yet!).\n"
            raise NotImplementedError(err)
        
        self._num_threads = num_threads

        # Finalize initialization
        self._success = None
        self._fit_type = "bayesian"

    def _setup_priors(self):
        """
        Set up the priors for the calculation.
        """

        # Create prior distribution to use for all gaussian prior calcs
        self._prior_frozen_rv = stats.norm(loc=0,scale=1)

        # Figure out the offset that normalizes the area of the pdf curve to 1.0
        # given the float precision etc. of the system
        base_offset = _find_normalization(scale=1,rv=stats.norm)

        uniform_priors = []
        gauss_prior_means = []
        gauss_prior_stds = []
        gauss_prior_offsets = []
        gauss_prior_mask = []

        for param in self.param_df.index:

            prior_mean = self.param_df.loc[param,"prior_mean"]
            prior_std = self.param_df.loc[param,"prior_std"]

            lower_bound = self.param_df.loc[param,"lower_bound"]
            upper_bound = self.param_df.loc[param,"upper_bound"]
            bounds = np.array([lower_bound,upper_bound])

            if np.isnan(prior_mean) or np.isnan(prior_std):
                uniform_priors.append(_find_uniform_value(bounds))
                gauss_prior_mask.append(False)

            else:
                gauss_prior_means.append(prior_mean)
                gauss_prior_stds.append(prior_std)

                z_bounds = (bounds - prior_mean)/prior_std
                bounds_offset = _reconcile_bounds_and_priors(bounds=z_bounds,
                                                             frozen_rv=self._prior_frozen_rv)
                gauss_prior_offsets.append(base_offset + bounds_offset)
                gauss_prior_mask.append(True)
    
        self._uniform_priors = np.sum(uniform_priors)

        self._gauss_prior_means = np.array(gauss_prior_means,dtype=float)
        self._gauss_prior_stds = np.array(gauss_prior_stds,dtype=float)
        self._gauss_prior_offsets = np.array(gauss_prior_offsets,dtype=float)
        self._gauss_prior_mask = np.array(gauss_prior_mask,dtype=bool)

        # Grab lower and upper bounds
        self._lower_bounds = np.array(self.param_df["lower_bound"])
        self._upper_bounds = np.array(self.param_df["upper_bound"])

    def _ln_prior(self,param):
        """
        Private function that gets the log prior without error checking. 
        """

        # If any parameter falls outside of the bounds, make the prior -infinity
        if np.sum(param < self._lower_bounds) > 0:
            return -np.inf
        
        if np.sum(param > self._upper_bounds) > 0:
            return -np.inf

        # Get priors for parameters we're treating with gaussian priors
        z = (param[self._gauss_prior_mask] - self._gauss_prior_means)/self._gauss_prior_stds
        gauss = np.sum(self._prior_frozen_rv.logpdf(z) + self._gauss_prior_offsets)

        # Return total priors
        return self._uniform_priors + gauss

    def ln_prior(self,param):
        """
        Log prior of fit parameters.  

        Parameters
        ----------
        param : numpy.ndarray
            float array of parameters to fit

        Returns
        -------
        prior : float
            log of priors.
        """

        self._sanity_check("fit can be done",["priors","bounds"])     

        param = check_array(value=param,
                            variable_name="param",
                            expected_shape=(self.num_params,),
                            expected_shape_names="(num_param,)")
        
        # This call should finalize the number of parameters if not already set
        if self.num_params is None:
            self._num_params = len(param)

        return self._ln_prior(param)


    def _ln_prob(self,param):
        """
        Private function that gets log probability without error checking.
        """

        # log posterior is log prior plus log likelihood
        ln_prob = self._ln_prior(param) + self._ln_like(param)

        # If result is not finite, this solution has an -infinity log
        # probability
        if not np.isfinite(ln_prob):
            return -np.inf

        return ln_prob

    def ln_prob(self,param):
        """
        Posterior probability of model parameters.

        Parameters
        ----------
        param : array of floats
            parameters to fit

        Returns
        -------
        ln_prob : float
            log posterior proability
        """

        self._sanity_check("fit can be done",["model","y_obs","y_stdev","priors","bounds"])
        
        param = check_array(value=param,
                            variable_name="param",
                            expected_shape=(self.num_params,),
                            expected_shape_names="(num_param,)")
        
        # This call should finalize the number of parameters if not already set
        if self.num_params is None:
            self._num_params = len(param)

        return self._ln_prob(param)

    def _fit(self,**kwargs):
        """
        Fit the parameters.

        Parameters
        ----------
        kwargs : dict
            keyword arguments to pass to emcee.EnsembleSampler
        """

        # Set up the priors
        self._setup_priors()

        guesses = np.array(self.param_df["guess"])            
        bounds = np.array([self._model.param_df["lower_bound"],
                            self._model.param_df["upper_bound"]])

        # Make initial guess (ML or just whatever the parameters sent in were)
        if self._ml_guess:
            fn = lambda *args: -self.weighted_residuals(*args)
            ml_fit = optimize.least_squares(fn,x0=guesses,bounds=bounds)
            self._initial_guess = np.copy(ml_fit.x)
        else:
            self._initial_guess = np.copy(guesses)

        # Create walker positions

        # Size of perturbation in parameter depends on the scale of the parameter
        perturb_size = self._initial_guess*self._initial_walker_spread

        ndim = len(guesses)
        pos = [self._initial_guess + np.random.randn(ndim)*perturb_size
               for i in range(self._num_walkers)]

        # Sample using walkers
        self._fit_result = emcee.EnsembleSampler(self._num_walkers,
                                                 ndim,
                                                 self._ln_prob,
                                                 **kwargs)

        self._fit_result.run_mcmc(pos, self._num_steps,progress=True)

        # Create list of samples
        to_discard = int(round(self._burn_in*self._num_steps,0))
        new_samples = self._fit_result.get_chain()[to_discard:,:,:].reshape((-1,ndim))

        if self.samples is None:
            self._samples = new_samples

        # If samples have already been done, append to them.
        else:
            self._samples = np.concatenate((self._samples,new_samples))

        self._lnprob = self._fit_result.get_log_prob()[:,:].reshape(-1)

        self._update_fit_df()

    def _update_fit_df(self):
        """
        Update samples based on the samples array.
        """

        # Get mean and standard deviation
        estimate = np.mean(self._samples,axis=0)
        std = np.std(self._samples,axis=0)

        # Calculate 95% confidence intervals
        lower = int(round(0.025*self._samples.shape[0],0))
        upper = int(round(0.975*self._samples.shape[0],0))
        low_95 = []
        high_95 = []
        for i in range(self._samples.shape[1]):
            nf = np.sort(self._samples[:,i])
            low_95.append(nf[lower])
            high_95.append(nf[upper])

        self._fit_df["estimate"] = estimate
        self._fit_df["std"] = std
        self._fit_df["low_95"] = low_95
        self._fit_df["high_95"] = high_95

        self._success = True

    @property
    def fit_info(self):
        """
        Information about the Bayesian run.
        """

        output = {}
        output["Num walkers"] = self._num_walkers
        output["Initial walker spread"] = self._initial_walker_spread
        output["Use ML guess"] = self._ml_guess
        output["Num steps"] = self._num_steps
        output["Burn in"] = self._burn_in

        if self.samples is not None:
            num_samples = len(self.samples[:,0])
        else:
            num_samples = None

        output["Final sample number"] = num_samples
        
        output["Num threads"] = self._num_threads

        return output
    
    def __repr__(self):

        out = ["BayesianSampler\n---------------\n"]

        out.append("Sampler parameters:\n")
        for k in self.fit_info:
            out.append(f"  {k}: {self.fit_info[k]}")

        out.append(f"\nanalysis has been run: {self._fit_has_been_run}\n")

        if self._fit_has_been_run:
            out.append(f"analysis results:\n")
            if self.success:
                for dataframe_line in repr(self.fit_df).split("\n"):
                    out.append(f"  {dataframe_line}")
                out.append("\n")
            else:
                out.append("  analysis failed\n")

        return "\n".join(out)