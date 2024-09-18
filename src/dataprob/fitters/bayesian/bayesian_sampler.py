"""
Fitter subclass for performing bayesian (MCMC) parameter estimation.
"""

from dataprob.fitters.base import Fitter
from dataprob.fitters.ml import MLFitter

from dataprob.fitters.bayesian._prior_processing import find_normalization
from dataprob.fitters.bayesian._prior_processing import find_uniform_value
from dataprob.fitters.bayesian._prior_processing import reconcile_bounds_and_priors
from dataprob.fitters.bayesian._prior_processing import create_walkers

from dataprob.util.check import check_int
from dataprob.util.check import check_float
from dataprob.util.check import check_bool
from dataprob.util.check import check_array

from dataprob.util.stats import get_kde_max

import emcee

import numpy as np
from scipy import stats

import multiprocessing
import sys
import warnings



class BayesianSampler(Fitter):
    """
    Use Bayesian MCMC to sample parameter space. 
    """

    def _setup_priors(self):
        """
        Set up the priors for the calculation.
        """

        # Create prior distribution to use for all gaussian prior calcs
        self._prior_frozen_rv = stats.norm(loc=0,scale=1)

        # Figure out the offset that normalizes the area of the pdf curve to 1.0
        # given the float precision etc. of the system
        base_offset = find_normalization(scale=1,rv=stats.norm)

        uniform_priors = []
        gauss_prior_means = []
        gauss_prior_stds = []
        gauss_prior_offsets = []
        gauss_prior_mask = []

        for param in self.param_df.index:

            # If a parameter is fixed, ignore it completely here. The param
            # array that comes in will not have an entry for this parameter 
            # so it should not even be in the mask as a False
            if self.param_df.loc[param,"fixed"]:
                continue

            # Get prior mean and std. 
            prior_mean = self.param_df.loc[param,"prior_mean"]
            prior_std = self.param_df.loc[param,"prior_std"]

            # Get bounds
            lower_bound = self.param_df.loc[param,"lower_bound"]
            upper_bound = self.param_df.loc[param,"upper_bound"]
            bounds = np.array([lower_bound,upper_bound])

            # If prior_mean or prior_std is nan, use uniform priors. 
            if np.isnan(prior_mean) or np.isnan(prior_std):

                # Set the gauss_prior_mask to False for this parameter and add
                # an appropriate chunk to the uniform prior 
                gauss_prior_mask.append(False)
                uniform_priors.append(find_uniform_value(bounds))
        
            else:

                # Set gauss_prior_mask to True for this parameter
                gauss_prior_mask.append(True)

                # Record gauss prior mean and std for use in the on-the-fly
                # prior calc
                gauss_prior_means.append(prior_mean)
                gauss_prior_stds.append(prior_std)

                # Reconcile the bounds and priors to find the normalization 
                # offset for this parameter
                z_bounds = (bounds - prior_mean)/prior_std
                bounds_offset = reconcile_bounds_and_priors(bounds=z_bounds,
                                                            frozen_rv=self._prior_frozen_rv)
                
                # Record the normalization offset
                gauss_prior_offsets.append(base_offset + bounds_offset)
                
    
        self._uniform_priors = np.sum(uniform_priors)

        self._gauss_prior_means = np.array(gauss_prior_means,dtype=float)
        self._gauss_prior_stds = np.array(gauss_prior_stds,dtype=float)
        self._gauss_prior_offsets = np.array(gauss_prior_offsets,dtype=float)
        self._gauss_prior_mask = np.array(gauss_prior_mask,dtype=bool)

        # Grab lower and upper bounds. We pull them out of the dataframe so we
        # can use in prior calculations without any dictionary lookups. 
        unfixed = self._model.unfixed_mask
        self._lower_bounds = np.array(self.param_df.loc[unfixed,"lower_bound"],
                                      dtype=float).copy()
        self._upper_bounds = np.array(self.param_df.loc[unfixed,"upper_bound"],
                                      dtype=float).copy()

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

    def _sample_to_convergence(self):
        """
        Run the sampler up to _max_convergence_cycles times in an effort go 
        get converged parameter estimates. This is based on emcee's estimated
        autocorrelation time for the parameters. The convergence criterion is
        50x the longest autocorrelation time. 
        """

        # Print status to standard error (like the progress bars)
        print(f"Running 1 of up to {self._max_convergence_cycles} sampler iterations",
            flush=True,
            file=sys.stderr)

        # Run initial sampler pass
        self._fit_result.run_mcmc(initial_state=self._initial_state,
                                  nsteps=self._num_steps,
                                  progress=True)

        # Initialize control variables
        success = False
        counter = 1

        # Loop until we reach max_convergence_cycles
        while counter < self._max_convergence_cycles:
            
            # Get the parameter correlation time. 
            try:

                # Get largest correlation time--this limits our final sampling. If 
                # this runs without error, break out of the loop and declare 
                # success. 
                max_corr = np.max(self._fit_result.get_autocorr_time())
                print(f"   Converged correlation time: {max_corr:.2f} iterations\n",
                      flush=True,
                      file=sys.stderr)
                success = True
                break

            except emcee.autocorr.AutocorrError as e:

                # emcee throws an AutocorrError if it is not converged yet. Get the
                # maximum of its current (unreliable) guess. 
                max_corr = np.max(e.__dict__["tau"])

                # Figure out how many steps to add to get to the target correlation time
                # (max_corr * 50). This number is hard-coded into the acorr estimator. 
                need_at_least = int(np.ceil(max_corr)*50)

                msg = f"   Rough estimate of correlation time: {max_corr:.2f} iterations. "
                msg +=f"We need at least {need_at_least} samples.\n"
                print(msg,flush=True,file=sys.stderr)
                
            # Run next sampling, starting from previous state
            print(f"Running {counter + 1} of up to {self._max_convergence_cycles} sampler iterations",
                  flush=True,
                  file=sys.stderr)
            
            # Current number of steps
            current_num_steps = self._fit_result.get_chain().shape[0]
            next_iteration = need_at_least - current_num_steps

            self._fit_result.run_mcmc(initial_state=None,
                                      nsteps=next_iteration,
                                      progress=True)

            # Update the counter
            counter += 1

        # Final number of steps taken
        num_steps = self._fit_result.get_chain().shape[0]

        # If we converged, write this message out
        if success:
            print(f"\nTook {num_steps} steps ({num_steps/max_corr:.1f}x the correlation time)\n",
                flush=True,
                file=sys.stderr)
            
        # If we failed to converge, warn about this. 
        else:
            w = f"\n\nParameter correlation time did not converge after "
            w += f"{self._max_convergence_cycles} cycles.\n"
            w += "Try increasing max_convergence_cycles when calling fit().\n\n"
            warnings.warn(w)

        # Record success (or not)
        self._success = success


    def fit(self,
            y_obs=None,
            y_std=None,
            num_walkers=100,
            use_ml_guess=True,
            num_steps=100,
            burn_in=0.1,
            num_threads=1,
            max_convergence_cycles=1,
            **emcee_kwargs):
        """
        Perform Bayesian MCMC sampling of parameter values. 

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
        num_walkers : int, default=100
            number of markov chains to use in the analysis
        use_ml_guess : bool, default=True
            if true, do a maximum likelihood maximization then sample from the
            fit parameter covariance matrix to get the initial chain positions
        num_steps: int, default=100
            number of steps to run each markov chain
        burn_in : float, default = 0.1
            fraction of samples to discard from the start of the run
        num_threads : int, default=1
            number of threads to use.  if `0`, use the total number of cpus. 
            [NOT YET IMPLEMENTED]
        max_convergence_cycles : int, default=1
            run this number of cycles in an attempt to get the parameter 
            estimates to converge. Must be >= 1. convergence is detected by 
            estimating the parameter autocorrelation time (in units of steps),
            then aiming for 50 times more steps than the autocorrelation time. 
        **emcee_kwargs : 
            all remaining keyword arguments are passed to the initialization 
            function of emcee.EnsembleSampler
        """
        
        # Set keywords, validating as we go
        self._num_walkers = check_int(value=num_walkers,
                                      variable_name="num_walkers",
                                      minimum_allowed=1)    
        self._use_ml_guess = check_bool(value=use_ml_guess,
                                        variable_name="use_ml_guess")
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

        # max convergence time
        self._max_convergence_cycles = check_int(value=max_convergence_cycles,
                                                 variable_name="max_convergence_cycles",
                                                 minimum_allowed=1)

        super().fit(y_obs=y_obs,
                    y_std=y_std,
                    **emcee_kwargs)     

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

        # Construct initial walker positions. If use_ml_guess is specified, do a
        # maximum likelihood fit, then sample from the fit parameter covariance
        # matrix to generate initial guesses. This will sample only unfixed
        # parameters. 
        if self._use_ml_guess:

            ml_fit = MLFitter(some_function=self._model)
            ml_fit.param_df = self.param_df.copy()
            ml_fit.data_df = self.data_df.copy()

            try:
                ml_fit.fit(num_samples=self._num_walkers*100)
            except Exception as e:
                err = "\n\nInitial ml fit is failing. See error trace for details.\n\n"
                raise RuntimeError(err) from e
            
            # Get samples
            success = False
            if ml_fit.samples is not None:
                self._initial_state = ml_fit.samples[:self._num_walkers,:]
                success = True

            # If we are not getting samples out of the ml fitter.
            if not success:
                err = "\n\nml fitter is not returning parameter samples. This can\n"
                err += "occur if the initial guesses are extremely far from\n"
                err += "true values or if the model is not numerically stable.\n"
                err += "Try changing your parameter guesses and/or parameter\n"
                err += "bounds. Alternatively, set use_ml_guess = False.\n\n"
                raise RuntimeError(err)

        # Generate walkers by sampling from the prior distribution. This will
        # only generate values for unfixed parameters. 
        else:
            self._initial_state = create_walkers(param_df=self.param_df,
                                                 num_walkers=self._num_walkers)
        
        # Build sampler object
        self._fit_result = emcee.EnsembleSampler(nwalkers=self._num_walkers,
                                                 ndim=self._initial_state.shape[1],
                                                 log_prob_fn=self._ln_prob,
                                                 **kwargs)

        # Run sampler
        self._sample_to_convergence()
    
        # Create numpy array of samples
        to_discard = int(round(self._burn_in*self._num_steps,0))
        chains = self._fit_result.get_chain()[to_discard:,:,:]
        new_samples = chains.reshape((-1,self._initial_state.shape[1]))

        # Create numpy array of lnprob for each sample
        new_lnprob = self._fit_result.get_log_prob()[:,:].reshape(-1)
        new_lnprob = new_lnprob[-new_samples.shape[0]:]

        if self.samples is None:
            self._samples = new_samples
            self._lnprob = new_lnprob
        else:
            self._samples = np.concatenate((self._samples,new_samples))
            self._lnprob = np.concatenate((self._lnprob,new_lnprob))

        self._update_fit_df()

    def _update_fit_df(self):
        """
        Update samples based on the samples array.
        """

        # Get mean and standard deviation
        estimate = get_kde_max(self._samples)
        std = np.std(self._samples,axis=0)

        # Calculate 95% confidence intervals
        lower = int(round(0.025*self._samples.shape[0],0))
        upper = int(round(0.975*self._samples.shape[0],0))

        # For samples less than ~100, the rounding above will make the
        # the upper cutoff the number of samples, and thus lead to an index 
        # error below. 
        if upper >= self._samples.shape[0]:
            upper = self._samples.shape[0] - 1

        low_95 = []
        high_95 = []
        for i in range(self._samples.shape[1]):
            sorted_samples = np.sort(self._samples[:,i])
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
        Information about the Bayesian run.
        """

        output = {}

        if hasattr(self,"_num_walkers"):
            output["Num walkers"] = self._num_walkers
        
        if hasattr(self,"_use_ml_guess"):
            output["Use ML guess"] = self._use_ml_guess
        
        if hasattr(self,"_num_steps"):
            output["Num steps"] = self._num_steps
        
        if hasattr(self,"_burn_in"):
            output["Burn in"] = self._burn_in
        
        if hasattr(self,"_max_convergence_cycles"):
            output["Max convergence cycles"] = self._max_convergence_cycles


        if self.samples is not None:
            num_samples = self.samples.shape[0]
        else:
            num_samples = None

        output["Final sample number"] = num_samples
        
        if hasattr(self,"_num_threads"):
            output["Num threads"] = self._num_threads

        return output
    
    def __repr__(self):
        """
        Output to show when object is printed or displayed in a jupyter 
        notebook.
        """

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