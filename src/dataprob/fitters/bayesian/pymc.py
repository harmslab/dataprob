"""
Fitter subclass for performing bayesian (MCMC) parameter estimation using PyMC.
"""
from ..base import Fitter
from ...util.check import check_int
from ...util.stats import get_kde_max
import numpy as np
import warnings
import traceback
import pandas as pd
import os
import csv

try:
    import pymc as pm
    import pytensor
    import pytensor.tensor as pt
    from pytensor.graph.op import Op
    from pytensor.gradient import grad_not_implemented
    PYMC_AVAILABLE = True
except ImportError:
    PYMC_AVAILABLE = False

class PyMCFitter(Fitter):
    """
    Use Bayesian MCMC via the PyMC library to sample parameter space.
    """

    def __init__(self, *args, **kwargs):
        if not PYMC_AVAILABLE:
            raise ImportError("PyMC is not installed. Please install it via 'pip install pymc'")
        super().__init__(*args, **kwargs)

    def _fit(self, draws=1000, tune=1000, chains=4, target_accept=0.9, output_dir=None, **pymc_kwargs):
        """Perform MCMC sampling using PyMC.
        
        Parameters
        ----------
        draws : int
            Number of samples to draw per chain
        tune : int
            Number of tuning steps
        chains : int
            Number of MCMC chains
        target_accept : float
            Target acceptance rate for NUTS sampler
        output_dir : str, optional
            Directory to save MCMC results. Creates three CSV files:
            - fit_results.csv: Parameter estimates
            - samples.csv: MCMC samples (thinned)
            - fit_summary.csv: Diagnostics (R-hat, ESS, etc.)
        **pymc_kwargs : dict
            Additional arguments passed to pm.sample()
        """
        self._draws = check_int(draws, "draws", 1)
        self._tune = check_int(tune, "tune", 1)
        self._chains = check_int(chains, "chains", 1)
        
        # Validate output_dir
        if output_dir is not None and not isinstance(output_dir, str):
            raise TypeError("output_dir must be a string or None")
        self._output_dir = output_dir

        class NumpyModelAndJacOp(Op):
            """
            Combined Op that computes y_hat and the full Jacobian J together in a
            single perform() call. Returning both as outputs guarantees that the
            expensive equilibrium solve (inside model_function / jacobian_function)
            happens exactly once per concrete parameter point, whether PyMC is
            evaluating the forward pass or the gradient.

            Caching requirement
            -------------------
            This Op calls model_function(params) and then jacobian_function(params)
            in sequence. To avoid a redundant second forward pass, jacobian_function
            must cache the result of the most recent model_function call and reuse
            it when called with the same params. If jacobian_function does not
            implement this caching, the forward pass will be executed twice per
            gradient evaluation, which is correct but slower.

            Outputs
            -------
            [0] y_hat : dvector  (N_obs,)
            [1] J     : dmatrix  (N_obs, N_params)
            """
            itypes = [pt.dvector]
            otypes = [pt.dvector, pt.dmatrix]

            def __init__(self, model_function, jacobian_function,
                         non_fit_kwargs, y_obs_len, param_len):
                self.model_function   = model_function
                self.jacobian_function = jacobian_function
                self.non_fit_kwargs   = non_fit_kwargs
                self.y_obs_len        = y_obs_len
                self.param_len        = param_len

            def perform(self, node, inputs, output_storage):
                params = inputs[0]

                # ── forward pass ────────────────────────────────────────────
                try:
                    y_hat = self.model_function(params, **self.non_fit_kwargs)
                    if y_hat is None or y_hat.shape[0] != self.y_obs_len:
                        y_hat = np.full(self.y_obs_len, np.nan)
                except Exception:
                    y_hat = np.full(self.y_obs_len, np.nan)

                # ── Jacobian ────────────────────────────────────────────────
                # jacobian_normalized() caches its forward solve internally;
                # because model_function (= model_normalized) was just called
                # with the same params, the cache always hits here.
                try:
                    J = self.jacobian_function(params)
                    if J is None or not np.all(np.isfinite(J)):
                        J = np.zeros((self.y_obs_len, self.param_len))
                except Exception:
                    J = np.zeros((self.y_obs_len, self.param_len))

                output_storage[0][0] = np.asarray(y_hat, dtype='float64')
                output_storage[1][0] = np.asarray(J,     dtype='float64')

            def grad(self, inputs, output_grads):
                # g_y_hat: upstream gradient of the scalar loss w.r.t. y_hat output
                g_y_hat = output_grads[0]

                # Re-apply this Op to get a graph node whose outputs[1] is J_sym.
                # PyTensor CSE will deduplicate this node with the forward node when
                # the same params tensor is used, so perform() is called only once.
                y_hat_sym, J_sym = self(inputs[0])

                # Vector-Jacobian product: (N_obs,) · (N_obs × N_params) → (N_params,)
                # This gives d(loss)/d(params) = g_y_hat · J
                g_params = pt.dot(g_y_hat, J_sym)
                return [g_params]


        class NumpyModelOp(Op):
            """
            Minimal Op for the gradient-free Slice-sampler fallback.
            This Op is only used when no Jacobian is available.
            """
            itypes = [pt.dvector]
            otypes = [pt.dvector]

            def __init__(self, model_function, non_fit_kwargs, y_obs_len):
                self.model_function    = model_function
                self.non_fit_kwargs    = non_fit_kwargs
                self.y_obs_len         = y_obs_len

            def perform(self, node, inputs, output_storage):
                params = inputs[0]
                try:
                    result = self.model_function(params, **self.non_fit_kwargs)
                    if result is None or result.shape[0] != self.y_obs_len:
                        result = np.full(self.y_obs_len, np.nan)
                except Exception:
                    result = np.full(self.y_obs_len, np.nan)
                output_storage[0][0] = np.asarray(result, dtype='float64')

            def grad(self, inputs, output_grads):
                # No Jacobian available, so gradient is not implemented.
                return [grad_not_implemented(self, 0, inputs[0])]
        
        has_jacobian = False
        jacobian_function = None
        
        fit_func = self._model._model_to_fit
        if hasattr(fit_func, "__self__"):
            original_object = fit_func.__self__
            if hasattr(original_object, "jacobian_normalized") and callable(original_object.jacobian_normalized):
                has_jacobian = True
                jacobian_function = original_object.jacobian_normalized

        with pm.Model() as model:
            params = {}
            unfixed_param_names = []
            for p_name in self.param_df.index:
                p_info = self.param_df.loc[p_name]
                if p_info["fixed"]:
                    params[p_name] = pt.as_tensor_variable(p_info["guess"])
                    continue
                
                unfixed_param_names.append(p_name)
                prior_mean, prior_std = p_info["prior_mean"], p_info["prior_std"]
                lower, upper = p_info["lower_bound"], p_info["upper_bound"]

                if not np.isnan(prior_mean) and not np.isnan(prior_std):
                    params[p_name] = pm.TruncatedNormal(p_name, mu=prior_mean, sigma=prior_std, lower=lower, upper=upper)
                else:
                    if np.isinf(lower) or np.isinf(upper):
                        raise ValueError(f"PyMC requires finite bounds for Uniform priors. Check parameter '{p_name}'.")
                    params[p_name] = pm.Uniform(p_name, lower=lower, upper=upper)

            all_params_in_order = [params[p] for p in self.param_df.index]
            full_symbolic_vector = pt.stack(all_params_in_order)

            if has_jacobian:
                # Use the combined Op when Jacobian is available
                combined_op = NumpyModelAndJacOp(
                    model_function=self._model._model_to_fit,
                    jacobian_function=jacobian_function,
                    non_fit_kwargs=self.non_fit_kwargs,
                    y_obs_len=len(self._y_obs),
                    param_len=len(self.param_df)
                )
                y_hat, _ = combined_op(full_symbolic_vector) # y_hat is the first output
            else:
                # Fallback to minimal Op for gradient-free sampling
                numpy_model_op = NumpyModelOp(
                    model_function=self._model._model_to_fit,
                    non_fit_kwargs=self.non_fit_kwargs,
                    y_obs_len=len(self._y_obs)
                )
                y_hat = numpy_model_op(full_symbolic_vector)

            pm.Normal("obs", mu=y_hat, sigma=self._y_std, observed=self._y_obs)

            initvals = {
                p_name: self.param_df.loc[p_name, "guess"]
                for p_name in unfixed_param_names
            }
        
            # Disable concentration tracking during MCMC to avoid O(n²) DataFrame growth.
            # Each get_concs() call normally appends to a DataFrame via pd.concat;
            # with thousands of NUTS evaluations this causes severe slowdown.
            _bm_obj = None
            _orig_track = True
            if hasattr(fit_func, "__self__"):
                _global_model = fit_func.__self__
                if hasattr(_global_model, "_bm") and hasattr(_global_model._bm, "_track_concentrations"):
                    _bm_obj = _global_model._bm
                    _orig_track = _bm_obj._track_concentrations
                    _bm_obj._track_concentrations = False

            try:
                sampler_kwargs = pymc_kwargs.copy()
                if not has_jacobian and "step" not in sampler_kwargs:
                    print("WARNING: No analytical Jacobian available. Using the gradient-free Slice sampler. This may be slow.")
                    sampler_kwargs["step"] = pm.Slice()
                
                if has_jacobian:
                    print("INFO: Analytical Jacobian found. Using NUTS sampler.")

                # target_accept is NUTS-only; skip it when using the Slice sampler
                nuts_kwargs = {"target_accept": target_accept} if has_jacobian else {}
                self._fit_result = pm.sample(draws=self._draws,
                                             tune=self._tune,
                                             chains=self._chains,
                                             initvals=initvals,
                                             **nuts_kwargs,
                                             **sampler_kwargs)
                self._success = True

            except Exception as e:
                self._success = False
                print(f"PyMC sampling failed: {e}")
                return
            finally:
                # Restore concentration tracking
                if _bm_obj is not None:
                    _bm_obj._track_concentrations = _orig_track

        if self._success:
            unfixed_params = self.param_df.index[self._model.unfixed_mask]
            posterior = self._fit_result.posterior.stack(sample=("chain", "draw"))
            sample_list = [posterior[p].values for p in unfixed_params]
            self._samples = np.stack(sample_list, axis=1)

            self._lnprob = None

            self._update_fit_df()
            
            # Save output files if output_dir was provided
            if self._output_dir is not None:
                os.makedirs(self._output_dir, exist_ok=True)
                
                # 1. Save fit_results.csv
                fit_results_file = os.path.join(self._output_dir, "fit_results.csv")
                # Replace inf/-inf with strings so Excel doesn't misread them
                out_df = self._fit_df.copy()
                out_df = out_df.replace([np.inf], "inf").replace([-np.inf], "-inf")
                out_df.to_csv(fit_results_file, index=False)
                print(f"Fit results saved to: {fit_results_file}")
                
                # 2. Save samples.csv (thinned to max 10000 samples)
                samples_file = os.path.join(self._output_dir, "samples.csv")
                max_samples = min(10000, self._samples.shape[0])
                thin = max(1, self._samples.shape[0] // max_samples)
                thinned_samples = self._samples[::thin, :]
                
                # Create samples dataframe with parameter names
                samples_df = pd.DataFrame(thinned_samples, columns=unfixed_params)
                samples_df.to_csv(samples_file, index=False)
                print(f"MCMC samples saved to: {samples_file} ({thinned_samples.shape[0]} samples)")
                
                # 3. Save fit_summary.csv with diagnostics
                summary_file = os.path.join(self._output_dir, "fit_summary.csv")
                
                try:
                    import arviz as az
                    summary = az.summary(self._fit_result, var_names=list(unfixed_params))
                    
                    with open(summary_file, 'w', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(["metric", "value"])
                        writer.writerow(["backend", "PyMC"])
                        writer.writerow(["draws_per_chain", self._draws])
                        writer.writerow(["tune_steps", self._tune])
                        writer.writerow(["num_chains", self._chains])
                        writer.writerow(["total_samples", self._samples.shape[0]])
                        writer.writerow(["thinned_samples_saved", thinned_samples.shape[0]])
                        writer.writerow(["", ""])  # Blank row
                        writer.writerow(["parameter", "mean", "std", "r_hat", "ess_bulk", "ess_tail"])
                        
                        for param in unfixed_params:
                            if param in summary.index:
                                row_data = summary.loc[param]
                                writer.writerow([
                                    param,
                                    f"{row_data['mean']:.6e}",
                                    f"{row_data['sd']:.6e}",
                                    f"{row_data.get('r_hat', np.nan):.4f}",
                                    f"{row_data.get('ess_bulk', np.nan):.1f}",
                                    f"{row_data.get('ess_tail', np.nan):.1f}"
                                ])
                    
                    print(f"MCMC diagnostics saved to: {summary_file}")
                    
                except ImportError:
                    # Fallback if arviz not available
                    with open(summary_file, 'w', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(["metric", "value"])
                        writer.writerow(["backend", "PyMC"])
                        writer.writerow(["draws_per_chain", self._draws])
                        writer.writerow(["tune_steps", self._tune])
                        writer.writerow(["num_chains", self._chains])
                        writer.writerow(["total_samples", self._samples.shape[0]])
                        writer.writerow(["thinned_samples_saved", thinned_samples.shape[0]])
                        writer.writerow(["note", "Install arviz for detailed diagnostics"])
                    
                    print(f"Basic summary saved to: {summary_file}")
                    print("  (Install arviz for R-hat and ESS diagnostics)")

    def _update_fit_df(self):
        """Update fit_df with results from the PyMC samples."""
        if self._samples is None or len(self._samples) == 0:
            return

        estimate = get_kde_max(self._samples)
        std = np.std(self._samples, axis=0)
        low_95, high_95 = np.quantile(self._samples, [0.025, 0.975], axis=0)

        for col in ["guess", "fixed", "lower_bound", "upper_bound", "prior_mean", "prior_std"]:
            self._fit_df[col] = self.param_df[col]

        unfixed = ~np.array(self._fit_df["fixed"], dtype=bool)
        self._fit_df.loc[unfixed, "estimate"] = estimate
        self._fit_df.loc[~unfixed, "estimate"] = self._fit_df.loc[~unfixed, "guess"]
        self._fit_df.loc[unfixed, "std"] = std
        self._fit_df.loc[unfixed, "low_95"] = low_95
        self._fit_df.loc[unfixed, "high_95"] = high_95

        # Check for derived parameters
        fit_func = self._model._model_to_fit
        if hasattr(fit_func, "__self__"):
            model_instance = fit_func.__self__
            if hasattr(model_instance, "calculate_derived_params"):
                try:
                    derived_df = model_instance.calculate_derived_params(samples=self._samples)
                    if derived_df is not None:
                        self._fit_df = pd.concat([self._fit_df, derived_df])
                except Exception as e:
                    warnings.warn(f"Could not calculate derived parameters: {e}")

    @property
    def fit_info(self):
        """Information about the Bayesian run."""
        output = {"Backend": "PyMC"}
        if hasattr(self, "_draws"):
            output.update({
                "Draws": self._draws,
                "Tune steps": self._tune,
                "Num chains": self._chains
            })
        output["Final sample number"] = self.samples.shape[0] if self.samples is not None else None
        if hasattr(self, "_fit_result"):
            output["Steps taken"] = self._draws * self._chains
        return output