"""
Fitter subclass for performing bayesian (MCMC) parameter estimation using emcee.
"""

from ..base import Fitter
from ...fitters.ml import MLFitter
from ._prior_processing import find_normalization, find_uniform_value, reconcile_bounds_and_priors, create_walkers
from ...util.check import check_int, check_float, check_bool, check_array
from ...util.stats import get_kde_max

import emcee
import numpy as np
from scipy import stats
import sys
import warnings

class EmceeFitter(Fitter):
    """
    Use Bayesian MCMC via the emcee library to sample parameter space.
    """

    def _setup_priors(self):
        """
        Set up the priors for the calculation.
        """
        self._prior_frozen_rv = stats.norm(loc=0, scale=1)
        base_offset = find_normalization(scale=1, rv=stats.norm)

        uniform_priors, gauss_prior_means, gauss_prior_stds = [], [], []
        gauss_prior_offsets, gauss_prior_mask = [], []

        for param in self.param_df.index:
            if self.param_df.loc[param, "fixed"]:
                continue

            prior_mean = self.param_df.loc[param, "prior_mean"]
            prior_std = self.param_df.loc[param, "prior_std"]
            bounds = np.array([self.param_df.loc[param, "lower_bound"],
                               self.param_df.loc[param, "upper_bound"]])

            if np.isnan(prior_mean) or np.isnan(prior_std):
                gauss_prior_mask.append(False)
                uniform_priors.append(find_uniform_value(bounds))
            else:
                gauss_prior_mask.append(True)
                gauss_prior_means.append(prior_mean)
                gauss_prior_stds.append(prior_std)
                z_bounds = (bounds - prior_mean) / prior_std
                bounds_offset = reconcile_bounds_and_priors(bounds=z_bounds,
                                                            frozen_rv=self._prior_frozen_rv)
                gauss_prior_offsets.append(base_offset + bounds_offset)

        self._uniform_priors = np.sum(uniform_priors)
        self._gauss_prior_means = np.array(gauss_prior_means, dtype=float)
        self._gauss_prior_stds = np.array(gauss_prior_stds, dtype=float)
        self._gauss_prior_offsets = np.array(gauss_prior_offsets, dtype=float)
        self._gauss_prior_mask = np.array(gauss_prior_mask, dtype=bool)

        unfixed = self._model.unfixed_mask
        self._lower_bounds = self.param_df.loc[unfixed, "lower_bound"].values.copy()
        self._upper_bounds = self.param_df.loc[unfixed, "upper_bound"].values.copy()

    def _ln_prior(self, param):
        """Private: gets the log prior without error checking."""
        if np.any(param < self._lower_bounds) or np.any(param > self._upper_bounds):
            return -np.inf

        z = (param[self._gauss_prior_mask] - self._gauss_prior_means) / self._gauss_prior_stds
        gauss = np.sum(self._prior_frozen_rv.logpdf(z) + self._gauss_prior_offsets)
        return self._uniform_priors + gauss

    def ln_prior(self, param):
        """Log prior of fit parameters."""
        self._sanity_check("fit can be done", ["model"])
        self._setup_priors()
        param = check_array(param, "param", (self.num_params,))
        return self._ln_prior(param)

    def _ln_prob(self, param):
        """Private: gets log probability without error checking."""
        ln_prob = self._ln_prior(param) + self._ln_like(param)
        return -np.inf if not np.isfinite(ln_prob) else ln_prob

    def ln_prob(self, param):
        """Posterior probability of model parameters."""
        self._sanity_check("fit can be done", ["model", "y_obs", "y_std"])
        self._setup_priors()
        param = check_array(param, "param", (self.num_params,))
        return self._ln_prob(param)

    def _sample_to_convergence(self):
        """Run sampler until convergence criteria are met."""
        print(f"Running 1 of up to {self._max_convergence_cycles} sampler iterations", file=sys.stderr)
        self._fit_result.run_mcmc(self._initial_state, self._num_steps, progress=True)
        
        print(f"  Total steps: {self._fit_result.iteration}", file=sys.stderr)
        print(f"  Mean acceptance fraction: {np.mean(self._fit_result.acceptance_fraction):.3f}", file=sys.stderr)
        print(f"  Max log probability: {np.max(self._fit_result.get_log_prob()):.2f}", file=sys.stderr)

        self._success = False
        for counter in range(1, self._max_convergence_cycles):
            try:
                max_corr = np.max(self._fit_result.get_autocorr_time())
                print(f"   Converged. Autocorrelation time: {max_corr:.2f} steps\n", file=sys.stderr)
                self._success = True
                break
            except emcee.autocorr.AutocorrError as e:
                max_corr = np.max(e.tau)
                need_at_least = int(np.ceil(max_corr) * 50)
                print(f"   Convergence not met. Est. autocorrelation time: {max_corr:.2f} steps. Need ~{need_at_least} total samples.\n", file=sys.stderr)

            current_num_steps = self._fit_result.get_chain().shape[0]
            if current_num_steps >= need_at_least:
                print("   Already have enough samples. Checking convergence again.", file=sys.stderr)
                continue

            print(f"Running {counter + 1} of up to {self._max_convergence_cycles} sampler iterations", file=sys.stderr)
            self._fit_result.run_mcmc(None, need_at_least - current_num_steps, progress=True)

            print(f"  Total steps: {self._fit_result.iteration}", file=sys.stderr)
            print(f"  Mean acceptance fraction: {np.mean(self._fit_result.acceptance_fraction):.3f}", file=sys.stderr)
            print(f"  Max log probability: {np.max(self._fit_result.get_log_prob()):.2f}", file=sys.stderr)

        num_steps = self._fit_result.get_chain().shape[0]
        if self._success:
            print(f"\nTook {num_steps} steps ({num_steps/max_corr:.1f}x the correlation time)\n", file=sys.stderr)
        else:
            warnings.warn(f"\n\nParameter correlation time did not converge after {self._max_convergence_cycles} cycles.\nTry increasing max_convergence_cycles.\n\n")

    def _fit(self,
             num_walkers=100,
             use_ml_guess=True,
             num_steps=100,
             burn_in=0.1,
             num_threads=1,
             max_convergence_cycles=1,
             output_dir=None,
             **emcee_kwargs):
        """Perform Bayesian MCMC sampling of parameter values."""
        self._num_walkers = check_int(num_walkers, "num_walkers", 1)
        self._use_ml_guess = check_bool(use_ml_guess, "use_ml_guess")
        self._num_steps = check_int(num_steps, "num_steps", 1)
        self._burn_in = check_float(burn_in, "burn_in", 0, 1, False, False)
        self._max_convergence_cycles = check_int(max_convergence_cycles, "max_convergence_cycles", 1)
        if output_dir is not None and not isinstance(output_dir, str):
            raise TypeError("output_dir must be a string or None")
        self._output_dir = output_dir
        
        if num_threads != 1:
            warnings.warn("multithreading has not yet been implemented for emcee backend.")
        self._num_threads = check_int(num_threads, "num_threads", 1)
        
        self._setup_priors()

        if self._use_ml_guess:
            ml_fit = MLFitter(self._model)
            ml_fit.param_df = self.param_df.copy()
            ml_fit.data_df = self.data_df.copy()
            try:
                ml_fit.fit(num_samples=self._num_walkers * 100)
                if ml_fit.samples is not None and len(ml_fit.samples) > self._num_walkers:
                    self._initial_state = ml_fit.samples[:self._num_walkers, :]
                else:
                    raise RuntimeError
            except Exception as e:
                err = "\n\nInitial ML fit failed or did not produce enough samples. Try changing guesses/bounds or set use_ml_guess=False.\n\n"
                raise RuntimeError(err) from e
        else:
            self._initial_state = create_walkers(self.param_df, self._num_walkers)

        self._fit_result = emcee.EnsembleSampler(self._num_walkers, self._initial_state.shape[1], self._ln_prob, **emcee_kwargs)

        try:
            self._sample_to_convergence()
        except KeyboardInterrupt:
            print("Sampling interrupted by user. Capturing last state.")
            self._success = False

        if self._fit_result.iteration > 0:
            to_discard = int(round(self._burn_in * self._fit_result.iteration, 0))
            chains = self._fit_result.get_chain(discard=to_discard)
            if chains.shape[0] > 0:
                self._samples = chains.reshape((-1, self._initial_state.shape[1]))
                self._lnprob = self._fit_result.get_log_prob(discard=to_discard).reshape(-1)
                self._update_fit_df()

                if self._output_dir is not None:
                    import os
                    import pandas as pd
                    os.makedirs(self._output_dir, exist_ok=True)

                    out_df = self._fit_df.copy()
                    out_df = out_df.replace([np.inf], "inf").replace([-np.inf], "-inf")
                    out_df.to_csv(os.path.join(self._output_dir, "fit_results.csv"))
                    print(f"Fit results saved to: {os.path.join(self._output_dir, 'fit_results.csv')}")

                    max_samples = min(10000, self._samples.shape[0])
                    thin = max(1, self._samples.shape[0] // max_samples)
                    thinned = self._samples[::thin, :]
                    pd.DataFrame(thinned, columns=self.param_df.index[~self.param_df["fixed"]]).to_csv(
                        os.path.join(self._output_dir, "samples.csv"), index=False)
                    print(f"MCMC samples saved to: {os.path.join(self._output_dir, 'samples.csv')} ({thinned.shape[0]} samples)")

    def _update_fit_df(self):
        """Update fit_df with results from the emcee samples."""
        if self.samples is None or len(self.samples) == 0:
            return

        estimate = get_kde_max(self._samples)
        std = np.std(self._samples, axis=0)
        
        lower_idx = int(round(0.025 * self.samples.shape[0], 0))
        upper_idx = int(round(0.975 * self.samples.shape[0], 0))
        if upper_idx >= self.samples.shape[0]:
            upper_idx = self.samples.shape[0] - 1
        
        low_95, high_95 = [], []
        for i in range(self.samples.shape[1]):
            sorted_samples = np.sort(self.samples[:, i])
            low_95.append(sorted_samples[lower_idx])
            high_95.append(sorted_samples[upper_idx])

        for col in ["guess", "fixed", "lower_bound", "upper_bound", "prior_mean", "prior_std"]:
            self._fit_df[col] = self.param_df[col]

        unfixed = ~np.array(self._fit_df["fixed"], dtype=bool)
        self._fit_df.loc[unfixed, "estimate"] = estimate
        self._fit_df.loc[~unfixed, "estimate"] = self._fit_df.loc[~unfixed, "guess"]
        self._fit_df.loc[unfixed, "std"] = std
        self._fit_df.loc[unfixed, "low_95"] = low_95
        self._fit_df.loc[unfixed, "high_95"] = high_95

    @property
    def fit_info(self):
        """Information about the Bayesian run."""
        output = {"Backend": "emcee"}
        if hasattr(self, "_num_walkers"):
            output.update({
                "Num walkers": self._num_walkers, "Use ML guess": self._use_ml_guess,
                "Num steps": self._num_steps, "Burn in": self._burn_in,
                "Max convergence cycles": self._max_convergence_cycles,
                "Num threads": self._num_threads
            })
        output["Final sample number"] = self.samples.shape[0] if self.samples is not None else None
        if hasattr(self, "_fit_result") and self._fit_result is not None:
            output["Steps taken"] = self._fit_result.iteration
        return output

    def __repr__(self):
        """
        Output to show when object is printed or displayed in a jupyter 
        notebook.
        """

        out = ["EmceeFitter\n-----------\n"]

        out.append(f"fit has been run: {self._fit_has_been_run}\n")
        if self._fit_has_been_run:
            out.append(f"fit results:\n")
            if self.success:
                status = "converged"
            else:
                status = "failed or interrupted"
            out.append(f"  fit status: {status}\n")

            # Always try to show the dataframe if it exists
            if hasattr(self, "_fit_df"):
                for dataframe_line in repr(self.fit_df).split("\n"):
                    out.append(f"  {dataframe_line}")
                out.append("\n")
            else:
                out.append("  fit dataframe not available\n")

        return "\n".join(out)