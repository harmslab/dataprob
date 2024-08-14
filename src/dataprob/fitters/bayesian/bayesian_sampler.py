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

        # Grab lower and upper bounds. We pull them out of the dataframe so we
        # can use in prior calculations without any dictionary lookups. 
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

        # Make sure model is loaded
        self._sanity_check("fit can be done",["model"])
        
        # Set up priors given model and param_df
        self._setup_priors()

        param = check_array(value=param,
                            variable_name="param",
                            expected_shape=(self.num_params,),
                            expected_shape_names="(num_param,)")
        
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

        self._sanity_check("fit can be done",["model","y_obs","y_std"])
        self._setup_priors()
        
        param = check_array(value=param,
                            variable_name="param",
                            expected_shape=(self.num_params,),
                            expected_shape_names="(num_param,)")
        
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

        to_fit = self._model.unfixed_mask
        guesses = np.array(self._model.param_df.loc[to_fit,"guess"])
        bounds = np.array([self._model.param_df.loc[to_fit,"lower_bound"],
                           self._model.param_df.loc[to_fit,"upper_bound"]])
        
        # Make initial guess (ML or just whatever the parameters sent in were)
        if self._ml_guess:
            def fn(*args): return -self._weighted_residuals(*args)
            ml_fit = optimize.least_squares(fn,x0=guesses,bounds=bounds)
            self._initial_guess = np.copy(ml_fit.x)
        else:
            self._initial_guess = np.copy(guesses)

        # Create walker positions

        # Size of perturbation in parameter depends on the scale of the parameter
        perturb_size = self._initial_guess*self._initial_walker_spread

        ndim = len(self._initial_guess)
        pos = [self._initial_guess + np.random.randn(ndim)*perturb_size
               for _ in range(self._num_walkers)]

        # Sample using walkers
        self._fit_result = emcee.EnsembleSampler(self._num_walkers,
                                                 ndim,
                                                 self._ln_prob,
                                                 **kwargs)

        self._fit_result.run_mcmc(pos, self._num_steps,progress=True)

        # Create numpy array of samples
        to_discard = int(round(self._burn_in*self._num_steps,0))
        new_samples = self._fit_result.get_chain()[to_discard:,:,:].reshape((-1,ndim))

        # Create numpy array of lnprob for each sample
        new_lnprob = self._fit_result.get_log_prob()[:,:].reshape(-1)
        new_lnprob = new_lnprob[-new_samples.shape[0]:]

        if self.samples is None:
            self._samples = new_samples
            self._lnprob = new_lnprob
        else:
            self._samples = np.concatenate((self._samples,new_samples))
            self._lnprob = np.concatenate((self._lnprob,new_lnprob))

        self._success = True

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
            num_samples = self.samples.shape[0]
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