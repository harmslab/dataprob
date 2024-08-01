"""
Fitter subclass for performing bayesian (MCMC) parameter estimation.
"""

from .base import Fitter

import emcee

import numpy as np
import scipy.optimize as optimize
from scipy import stats

import warnings
import multiprocessing

def _find_normalization(scale,rv,**kwargs):
    """
    This method finds an offset to add to the output of rv.logpdf that 
    makes the total area under the pdf curve 1.0. 

    First, calculate the unnormalized pdf at the center of the distribution
    (loc):

        un_norm_prob = rv.pdf(loc)

    Second, calculate the minimum difference between floats we can represent
    using our data type. This sets the bin width for a discrete approximation
    of the continuous distribution.

        res = np.finfo(un_norm_prob.dtype).resolution

    Third, calculate the difference in the cdf between loc + num_to_sample*res
    and loc - num_to_sample*res. This is the normalized probability of the slice
    centered on loc. 

        norm_prob = cdf(loc + num_to_sample*res) - cdf(loc - num_to_sample*res). 

    The ratio of norm_prob and un_norm_prob is a scalar that converts from raw
    pdf calculations to normalized probabilities. 

        norm_prob_x = rv.pdf(x)*norm_prob/un_norm_prob

    Since we care about log versions of this, it becomes:

        log_norm_prob_x = rv.logpdf(x) + log(norm_prob) - log(un_norm_prob)

    We can define a pre-calculated offset;

        offset = log(norm_prob) - log(un_norm_prob) 

    So, finally:

        log_norm_prob_x = rv.logpdf(x) + offset

    This is most numerically stable at loc = 0, so figure out the
    normalization using all other shape parameters but loc = 0. 
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

def _process_bounds(bounds,frozen_rv):

    # bounds specified, no offset to area
    if bounds is None:
        return 0.0

    # Parse bounds list
    try: 
        left = float(bounds[0])
        right = float(bounds[1])
    except Exception as e:
        err = "bounds must be an iterable with an upper and lower value\n"
        raise ValueError(err) from e

    # Check sanity of bounds
    if left > right:
        err = f"left bound '{bounds[0]}' must be smaller than right bound '{bounds[1]}'\n"
        raise ValueError(err)

    # left and right the same, infinite prior for the shared value
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
    # are identical within numerical error
    if remaining == 0:
        w = f"left and right bounds {bounds} are numerically identical\n"
        warnings.warn(w)
        return 0.0

    # Return amount to scale the area by
    return np.log(1/remaining)

def _find_uniform_value(bounds):

    finfo = np.finfo(np.ones(1,dtype=float)[0].dtype)
    
    # The probability of being one of the non-infinite points with step size
    # of resolution
    if bounds is None:
        return np.log(finfo.resolution) - (np.log(2) + np.log(finfo.max))

    # Parse bounds list
    try: 
        left = float(bounds[0])
        right = float(bounds[1])
    except Exception as e:
        err = "bounds must be an iterable with an upper and lower value\n"
        raise ValueError(err) from e

    # Check sanity of bounds
    if left > right:
        err = f"left bound '{bounds[0]}' must be smaller than right bound '{bounds[1]}'\n"
        raise ValueError(err)

    # left and right the same, infinite prior for the shared value
    if left == right:
        w = f"left and right bounds {bounds} are identical\n"
        warnings.warn(w)
        return 0
    
    # Probability of being one number with step size of resolution between the
    # edges specified by bounds
    return np.log(finfo.resolution) - np.log((bounds[1] - bounds[0]))



class BayesianFitter(Fitter):
    """
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
        num_threads : int or `"max"`
            number of threads to use.  if `"max"`, use the total number of
            cpus. [NOT YET IMPLEMENTED]
        """

        super().__init__()

        self._num_walkers = num_walkers
        self._initial_walker_spread = initial_walker_spread
        self._ml_guess = ml_guess
        self._num_steps = num_steps
        self._burn_in = burn_in

        self._num_threads = num_threads
        if self._num_threads == "max":
            self._num_threads = multiprocessing.cpu_count()

        if not type(self._num_threads) == int and self._num_threads > 0:
            err = "num_threads must be 'max' or a positive integer\n"
            raise ValueError(err)

        if self._num_threads != 1:
            err = "multithreading has not yet been implemented (yet!).\n"
            raise NotImplementedError(err)

        self._success = None

        self.fit_type = "bayesian"

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

        # If any parameter falls outside of the bounds, make the prior -infinity
        if np.sum(param < self.bounds[0,:]) > 0 or np.sum(param > self.bounds[1,:]) > 0:
            return -np.inf

        # Get priors for parameters we're treating with uniform priors
        uniform = np.sum(self._uniform_prior)

        # Get priors for parameters we're treating with gaussian priors
        z = (param - self._prior_means)/self._prior_stds
        gauss = np.sum(self._prior_frozen_rv.logpdf(z) + self._prior_gaussian_offsets)

        # Return total priors
        return uniform + gauss

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

        # Calculate prior.  
        ln_prior = self.ln_prior(param)

        # Calculate likelihood.  
        ln_like = self.ln_like(param)

        # log posterior is log prior plus log likelihood
        ln_prob = ln_prior + ln_like

        # If result is not finite, this solution has an -infinity log
        # probability
        if not np.isfinite(ln_prob):
            return -np.inf

        return ln_prob

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
        for i, s in enumerate(self._priors):

            bounds = self.bounds[i,:]

            if np.isnan(s[0]) or np.isnan(s[1]):
                uniform_priors.append(_find_uniform_value(bounds))

            else:
                gauss_prior_means.append(s[0])
                gauss_prior_stds.append(s[1])

                z_bounds = (bounds - s[0])/s[1]
                bounds_offset = _process_bounds(bounds=z_bounds,
                                                frozen_rv=self._prior_frozen_rv)
                gauss_prior_offsets.append(base_offset + bounds_offset)

    
        self._uniform_priors = np.array(uniform_priors,dtype=float)

        self._prior_means = np.array(gauss_prior_means,dtype=float)
        self._prior_stds = np.array(gauss_prior_stds,type=float)
        self._gauss_prior_offsets = np.array(gauss_prior_offsets)
        

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

        # Make initial guess (ML or just whatever the parameters sent in were)
        if self._ml_guess:
            fn = lambda *args: -self.weighted_residuals(*args)
            ml_fit = optimize.least_squares(fn,x0=self.guesses,bounds=self.bounds)
            self._initial_guess = np.copy(ml_fit.x)
        else:
            self._initial_guess = np.copy(self.guesses)

        # Create walker positions

        # Size of perturbation in parameter depends on the scale of the parameter
        perturb_size = self._initial_guess*self._initial_walker_spread

        ndim = len(self.guesses)
        pos = [self._initial_guess + np.random.randn(ndim)*perturb_size
               for i in range(self._num_walkers)]

        # Sample using walkers
        self._fit_result = emcee.EnsembleSampler(self._num_walkers,
                                                 ndim,
                                                 self.ln_prob,
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

        self._update_estimates()

    def _update_estimates(self):
        """
        Update samples based on the samples array.
        """

        # Get mean and standard deviation
        self._estimate = np.mean(self._samples,axis=0)
        self._stdev = np.std(self._samples,axis=0)

        # Calculate 95% confidence intervals
        self._ninetyfive = []
        lower = int(round(0.025*self._samples.shape[0],0))
        upper = int(round(0.975*self._samples.shape[0],0))
        self._ninetyfive = [[],[]]
        for i in range(self._samples.shape[1]):
            nf = np.sort(self._samples[:,i])
            self._ninetyfive[0].append(nf[lower])
            self._ninetyfive[1].append(nf[upper])

        self._ninetyfive = np.array(self._ninetyfive)

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
        output["Final sample number"] = len(self._samples[:,0])
        output["Num threads"] = self._num_threads

        return output
