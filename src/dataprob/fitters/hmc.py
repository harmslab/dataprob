"""
HMCFitter: Hamiltonian Monte Carlo sampler with explicit gradient injection.

Uses a self-contained leapfrog HMC implementation (pure numpy — no extra
dependencies beyond the rest of dataprob) with the symbolic Jacobian injected
as the explicit log-posterior gradient.  This gives full posterior samples —
including full parameter covariance — without requiring autodiff, and integrates
cleanly with the dataprob/linkage symbolic framework.

Algorithm
---------
Standard HMC with leapfrog integration and Metropolis-Hastings acceptance:

    1. Sample momentum p ~ N(0, I).
    2. Leapfrog for L steps of size eps:
           p_half = p      + (eps/2) * grad log pi(q)
           q      = q      + eps * p_half
           p      = p_half + (eps/2) * grad log pi(q)
    3. Accept q* with prob  min(1, exp( H(q,p) - H(q*,p*) ))
       where  H(q,p) = -log pi(q) + 0.5 * ||p||^2.

Gradient source
---------------
If the model object exposes ``jacobian_normalized(full_params)``, the analytic
symbolic Jacobian is used to build ∇ log π.  Otherwise a central finite-
difference fallback is applied with a warning.

Interface
---------
Follows the standalone IminuitFitter pattern: the model function is supplied at
construction, and y_obs / y_std / param_df are supplied at fit() time.
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
import scipy.optimize as spopt
import scipy.stats
import os
import csv
import time

from dataprob.fitters.base import Fitter
from dataprob.util.stats import get_kde_max


# ---------------------------------------------------------------------------
# Self-contained HMC engine
# ---------------------------------------------------------------------------

def _leapfrog(q, p, grad_fn, step_size, n_steps, M_inv=None):
    """Leapfrog integrator for HMC.  Returns (q*, p*) after n_steps steps."""
    q = q.copy()
    p = p.copy()
    # Half step for momentum
    p += 0.5 * step_size * grad_fn(q)
    for _ in range(n_steps - 1):
        q += step_size * (p if M_inv is None else M_inv @ p)
        p += step_size * grad_fn(q)
    q += step_size * (p if M_inv is None else M_inv @ p)
    # Final half step
    p += 0.5 * step_size * grad_fn(q)
    return q, p


def _hmc_sample(logprob_and_grad, x0, n_samples, step_size, n_steps, rng):
    """
    Draw *n_samples* from the distribution defined by logprob_and_grad.

    Parameters
    ----------
    logprob_and_grad : callable
        ``f(x) -> (logp, grad_logp)``  — both as floats/arrays.
    x0 : ndarray, shape (d,)
        Starting position.
    n_samples : int
        Number of HMC proposals (accepted or not).
    step_size : float
        Leapfrog step size eps.
    n_steps : int
        Leapfrog steps per proposal L.
    rng : np.random.Generator
        Random number generator.

    Returns
    -------
    samples : ndarray, shape (n_samples, d)
    n_accepted : int
    """
    d = x0.shape[0]
    samples   = np.empty((n_samples, d))
    q_current = x0.copy()

    lp_current, grad_current = logprob_and_grad(q_current)
    n_accepted = 0

    def grad_fn(q):
        _, g = logprob_and_grad(q)
        return g

    for i in range(n_samples):
        p_current = rng.standard_normal(d)

        # Hamiltonian at current state
        H_current = -lp_current + 0.5 * np.dot(p_current, p_current)

        # Leapfrog
        q_prop, p_prop = _leapfrog(q_current, p_current,
                                   grad_fn, step_size, n_steps)

        lp_prop, _ = logprob_and_grad(q_prop)
        H_prop = -lp_prop + 0.5 * np.dot(p_prop, p_prop)

        # MH acceptance
        log_accept = H_current - H_prop
        if np.log(rng.uniform()) < log_accept:
            q_current  = q_prop
            lp_current = lp_prop
            n_accepted += 1

        samples[i] = q_current

    return samples, n_accepted


# ---------------------------------------------------------------------------
# HMCFitter
# ---------------------------------------------------------------------------

class HMCFitter(Fitter):
    """
    Fit a model to data using Hamiltonian Monte Carlo with an optionally
    symbolic log-posterior gradient.

    Inherits from :class:`dataprob.fitters.base.Fitter` so it integrates
    with ``dataprob.setup()`` and shares the standard ``param_df``,
    ``data_df``, ``fit_df``, and ``samples`` interface.

    Parameters
    ----------
    some_function : callable
        The model function or bound method.  If the owning object exposes
        ``jacobian_normalized(full_params)``, it is used as the analytic
        gradient of the log-likelihood inside the HMC Hamiltonian.
    fit_parameters : list, dict, str, pandas.DataFrame; optional
        Passed to the base Fitter / ModelWrapper for parameter setup.
    non_fit_kwargs : dict, optional
        Fixed keyword arguments for some_function.
    vector_first_arg : bool, default=False
        If True, parameters are passed as a vector (first argument).
    """

    def __init__(self,
                 some_function,
                 fit_parameters=None,
                 non_fit_kwargs=None,
                 vector_first_arg=False):

        # Resolve the model object for Jacobian detection
        if hasattr(some_function, "__self__"):
            self._model_obj = some_function.__self__
        else:
            self._model_obj = some_function

        # Delegate parameter/model wrapping to the base Fitter
        super().__init__(some_function=some_function,
                         fit_parameters=fit_parameters,
                         non_fit_kwargs=non_fit_kwargs,
                         vector_first_arg=vector_first_arg)

        # Keep a direct callable reference for HMC internals
        self._model_fn = self._model.fast_model

        # Dict for pre-fit kwarg injection (e.g. fitter_kwargs['jacobian'])
        self.fitter_kwargs: dict = {}

        self._fit_result = None
        self._samples    = None

    # ------------------------------------------------------------------
    # Public fit API
    # ------------------------------------------------------------------

    def fit(self,
            y_obs=None,
            y_std=None,
            n_samples: int       = 2000,
            burn_in: int         = 400,
            step_size: float     = 0.1,
            n_steps: int         = 20,
            target_accept: float = 0.65,
            report_steps: int    = 100,
            random_seed: int | None = None,
            output_dir: str | None  = None,
            checkpoint_steps: int   = 0,
            resume_from: str | None = None,
            hessian_reg: float      = 1e-4,
            non_centered: bool      = False,
            **kwargs):
        """
        Sample the posterior of the model parameters using HMC.

        Parameters
        ----------
        y_obs : array-like
            Observed data values.
        y_std : array-like
            Standard deviations (uncertainties) on each observation.
        n_samples : int
            Total HMC proposals to draw (before burn-in removal). Default 2000.
        burn_in : int
            Number of leading samples to discard as burn-in. Default 400.
            During burn-in, dual averaging adapts step_size automatically.
        step_size : float
            Initial leapfrog step size (epsilon). Dual averaging will tune this
            during burn-in to hit target_accept. Default 0.1.
        n_steps : int
            Number of leapfrog steps per proposal L. Default 20.
        target_accept : float
            Target Metropolis acceptance rate for dual averaging. Default 0.65.
            Values in [0.5, 0.9] are typical; higher values → smaller step_size.
        report_steps : int
            Print a progress line every this many steps, for both burn-in
            and sampling phases. Must be a positive integer. Default 100.
        random_seed : int or None
            Seed for reproducibility. Default None (unseeded).
        output_dir : str, optional
            If provided, saves fit_summary.csv, fit_results.csv, and
            hmc_checkpoint.npz there.
        checkpoint_steps : int, optional
            Save a checkpoint every this many sampling steps (0 = only on
            interrupt). Requires output_dir. Default 0.
        resume_from : str, optional
            Path to a checkpoint file or directory containing
            hmc_checkpoint.npz. When provided, burn-in is skipped and
            sampling continues from the saved state. Default None.
        non_centered : bool
            If True, apply the non-centered parameterization for all parameters
            that have a Gaussian prior (``prior_mean`` and ``prior_std`` both
            set).  Instead of sampling ``theta`` directly, the sampler draws
            ``z ~ N(0, 1)`` and recovers ``theta = prior_mean + z * prior_std``
            at each step.  This removes the correlation between the location/
            scale hyper-parameters and the per-observation offsets that causes
            HMC to explore slowly when the likelihood is weak relative to the
            prior (the "funnel" geometry).  Parameters with only bounds and no
            Gaussian prior are unaffected.  Default False.
        **kwargs
            Ignored; present for forward-compatibility with the base Fitter
            interface.
        """
        # Merge any pre-set fitter_kwargs (e.g. jacobian injection)
        merged = {**self.fitter_kwargs, **kwargs}
        # (currently no extra kwargs are consumed, but kept for extensibility)

        # Release any samples held from a previous run before allocating new ones
        self._samples = None

        self._n_samples         = int(n_samples)
        self._step_size         = float(step_size)
        self._initial_step_size = float(step_size)
        self._n_steps           = int(n_steps)
        self._target_accept     = float(np.clip(target_accept, 0.01, 0.99))
        self._report_steps      = int(report_steps)
        self._random_seed       = random_seed
        self._output_dir        = output_dir
        self._checkpoint_steps  = int(checkpoint_steps)
        self._resume_from       = resume_from
        self._hessian_reg       = float(hessian_reg)
        self._non_centered      = bool(non_centered)

        if self._report_steps < 1:
            raise ValueError("report_steps must be >= 1.")

        self._burn_in = int(burn_in)
        if self._burn_in < 0:
            raise ValueError("burn_in must be >= 0.")

        # Use the base Fitter's fit() to load y_obs/y_std and call _fit()
        super().fit(y_obs=y_obs, y_std=y_std)

    # ------------------------------------------------------------------
    # Internal fit logic
    # ------------------------------------------------------------------

    def _fit(self, **kwargs):
        """Run HMC sampling (called by base Fitter.fit())."""

        param_df        = self.param_df
        to_fit_mask     = np.array(param_df["fixed"], dtype=bool)
        to_fit_mask     = ~to_fit_mask
        unfixed_names   = list(param_df.loc[to_fit_mask, "name"])
        unfixed_guesses = np.array(param_df.loc[to_fit_mask, "guess"],        dtype=float)
        lower_bounds    = np.array(param_df.loc[to_fit_mask, "lower_bound"],  dtype=float)
        upper_bounds    = np.array(param_df.loc[to_fit_mask, "upper_bound"],  dtype=float)
        to_fit_indices  = np.where(to_fit_mask)[0]

        n_obs     = len(self._y_obs)
        n_unfixed = len(unfixed_names)
        self._dof = n_obs - n_unfixed

        # ---- detect gradient source ------------------------------------
        has_jac = (hasattr(self._model_obj, "jacobian_normalized")
                   and callable(self._model_obj.jacobian_normalized))
        self._gradient_type = "symbolic" if has_jac else "finite-difference"
        if not has_jac:
            warnings.warn(
                "HMCFitter: model does not expose 'jacobian_normalized'. "
                "Falling back to forward finite-difference gradient.  "
                "This is slower and less accurate than the symbolic Jacobian."
            )

        # ---- prior parameters -----------------------------------------
        prior_means = np.array(param_df.loc[to_fit_mask, "prior_mean"], dtype=float)
        prior_stds  = np.array(param_df.loc[to_fit_mask, "prior_std"],  dtype=float)
        has_gauss   = np.isfinite(prior_means) & np.isfinite(prior_stds) & (prior_stds > 0)

        # ---- non-centered parameterization setup -----------------------
        # nc_mask selects which unfixed params get the NC transform.
        # Only params with a Gaussian prior are eligible: for those we sample
        # z ~ N(0,1) and recover theta = prior_mean + z * prior_std.
        # Params with only bounds (uniform prior) are left in direct space.
        nc_mask = has_gauss & self._non_centered

        if np.any(nc_mask):
            # Scales and shifts for the z -> theta transform (1 / identity for non-NC)
            nc_scales = np.where(nc_mask, prior_stds,  1.0)
            nc_locs   = np.where(nc_mask, prior_means, 0.0)

            def _z_to_theta(z):
                """Convert sampling coords z to model params theta."""
                return nc_locs + z * nc_scales

            # Transform initial guesses and bounds into z-space
            x0_sampling = unfixed_guesses.copy()
            x0_sampling[nc_mask] = ((unfixed_guesses[nc_mask] - prior_means[nc_mask])
                                    / prior_stds[nc_mask])
            s_lower = lower_bounds.copy()
            s_upper = upper_bounds.copy()
            s_lower[nc_mask] = ((lower_bounds[nc_mask] - prior_means[nc_mask])
                                / prior_stds[nc_mask])
            s_upper[nc_mask] = ((upper_bounds[nc_mask] - prior_means[nc_mask])
                                / prior_stds[nc_mask])
            nc_label = (f"non-centered ({int(nc_mask.sum())} of {n_unfixed} params)")
            print(f"  [non-centered] NC params: "
                  + ", ".join(n for n, m in zip(unfixed_names, nc_mask) if m))
        else:
            # Identity transform — sampling coords == model params
            _z_to_theta  = None
            x0_sampling  = unfixed_guesses.copy()
            s_lower      = lower_bounds.copy()
            s_upper      = upper_bounds.copy()
            nc_label     = "centered"

        # ---- full-param helper -----------------------------------------
        full_template = np.array(param_df["guess"], dtype=float)

        def _full(u):
            p = full_template.copy()
            p[to_fit_indices] = u
            return p

        # ---- log-prior and its gradient (in sampling / z-space) --------
        def _ln_prior(z):
            if np.any(z < s_lower) or np.any(z > s_upper):
                return -np.inf
            lp = 0.0
            if np.any(nc_mask):
                # NC params: prior absorbed into N(0,1) in z-space
                lp += float(np.sum(scipy.stats.norm.logpdf(z[nc_mask])))
                # Centered params that still have a Gaussian prior
                centered_gauss = has_gauss & ~nc_mask
                if np.any(centered_gauss):
                    dz = ((z[centered_gauss] - prior_means[centered_gauss])
                          / prior_stds[centered_gauss])
                    lp += float(np.sum(scipy.stats.norm.logpdf(dz)))
            elif np.any(has_gauss):
                dz  = (z[has_gauss] - prior_means[has_gauss]) / prior_stds[has_gauss]
                lp += float(np.sum(scipy.stats.norm.logpdf(dz)))
            return lp

        def _grad_ln_prior(z):
            g = np.zeros(n_unfixed)
            if np.any(nc_mask):
                # d/dz_i [ -0.5 z_i^2 ] = -z_i  for NC params
                g[nc_mask] = -z[nc_mask]
                centered_gauss = has_gauss & ~nc_mask
                if np.any(centered_gauss):
                    g[centered_gauss] = (-(z[centered_gauss] - prior_means[centered_gauss])
                                         / prior_stds[centered_gauss] ** 2)
            elif np.any(has_gauss):
                g[has_gauss] = (-(z[has_gauss] - prior_means[has_gauss])
                                / prior_stds[has_gauss] ** 2)
            return g

        # ---- finite-difference gradient fallback -----------------------
        _eps = 1e-5

        def _grad_ln_like_fd(y_calc_center, z):
            """Forward finite-difference gradient reusing the center y_calc.
            Perturbations are in sampling (z) space; each perturbed z is
            converted to theta before calling the model."""
            grad  = np.zeros(n_unfixed)
            sigma2 = self._y_std ** 2
            ll_c  = -0.5 * np.sum((y_calc_center - self._y_obs) ** 2 / sigma2)
            for i in range(n_unfixed):
                z2 = z.copy(); z2[i] += _eps
                theta2 = _z_to_theta(z2) if _z_to_theta is not None else z2
                try:
                    if hasattr(self._model_obj, "model_normalized") and callable(self._model_obj.model_normalized):
                        y2   = self._model_obj.model_normalized(_full(theta2))
                    else:
                        y2   = self._model.fast_model(_full(theta2))
                    ll2  = -0.5 * np.sum((y2 - self._y_obs) ** 2 / sigma2)
                    grad[i] = (ll2 - ll_c) / _eps
                except Exception:
                    pass
            return grad

        # ---- unified log-posterior + gradient (single model pass) ------
        # model_normalized sets _model_state_params; jacobian_normalized
        # checks that cache and skips the forward pass when params match,
        # giving one model evaluation per call instead of two.
        _model_error_shown = [False]

        def logprob_and_grad(z):
            lp = _ln_prior(z)
            if not np.isfinite(lp):
                # Outside bounds: repelling gradient toward interior (in z-space)
                grad = np.zeros(n_unfixed)
                for i in range(n_unfixed):
                    if z[i] < s_lower[i]:
                        grad[i] = s_lower[i] - z[i]
                    elif z[i] > s_upper[i]:
                        grad[i] = s_upper[i] - z[i]
                return -1e30, grad

            # Convert sampling coords to model params
            theta = _z_to_theta(z) if _z_to_theta is not None else z
            p = _full(theta)
            try:
                if hasattr(self._model_obj, "model_normalized") and callable(self._model_obj.model_normalized):
                    y_calc = self._model_obj.model_normalized(p)   # single forward pass
                else:
                    y_calc = self._model.fast_model(p)
            except Exception as _e:
                if not _model_error_shown[0]:
                    print(f"  [diagnostic] model exception: {type(_e).__name__}: {_e}")
                    _model_error_shown[0] = True
                return -1e30, np.zeros(n_unfixed)

            sigma2 = self._y_std ** 2
            ll     = -0.5 * np.sum((y_calc - self._y_obs) ** 2 / sigma2
                                   + np.log(2.0 * np.pi * sigma2))
            logp   = lp + ll
            if not np.isfinite(logp):
                return -1e30, np.zeros(n_unfixed)

            if has_jac:
                # jacobian_normalized reuses cached model state — no extra forward pass
                J_full  = self._model_obj.jacobian_normalized(p)        # (n_obs, n_full_params)
                J_uf    = J_full[:, to_fit_indices]                      # (n_obs, n_unfixed)
                # jacobian_normalized returns d(y_norm)/d(params) — singly normalised
                # by y_norm_std.  The likelihood residual r = (y_calc - y_obs)/y_std
                # is doubly normalised (y_norm_std AND y_std_scalar both folded
                # into y_std).  We must weight J by 1/y_std so the dot product
                # -J_w^T @ r uses a consistent normalisation on both sides.
                J_w_uf       = J_uf / self._y_std[:, np.newaxis]        # (n_obs, n_unfixed)
                grad_ll_theta = -J_w_uf.T @ (y_calc - self._y_obs)
                # Chain rule: d(logL)/dz_i = d(logL)/dtheta_i * dtheta_i/dz_i
                # dtheta_i/dz_i = nc_scales[i] (prior_std for NC params, 1 otherwise)
                grad_ll = (grad_ll_theta * nc_scales
                           if np.any(nc_mask) else grad_ll_theta)
            else:
                grad_ll = _grad_ln_like_fd(y_calc, z)

            return logp, grad_ll + _grad_ln_prior(z)

        # ---- output paths ----------------------------------------------
        fit_summary_file = None
        fit_results_file = None
        if self._output_dir is not None:
            os.makedirs(self._output_dir, exist_ok=True)
            fit_summary_file = os.path.join(self._output_dir, "fit_summary.csv")
            fit_results_file = os.path.join(self._output_dir, "fit_results.csv")
            print(f"HMCFitter: saving output to {self._output_dir}")

        # ---- run HMC ---------------------------------------------------
        rng = np.random.default_rng(self._random_seed)
        x0  = x0_sampling       # initial point in sampling (z) space
        d   = x0.shape[0]

        _adapt_note = (f", target_accept={self._target_accept:.2f} [dual-averaging]"
                       if self._burn_in > 0 else "")

        # ---- mass matrix setup -----------------------------------------
        has_hess = (hasattr(self._model_obj, 'hessian_normalized')
                    and callable(self._model_obj.hessian_normalized))
        _M_inv     = None
        _L         = None
        _mass_type = "identity"

        if has_hess:
            try:
                H0_full = self._model_obj.hessian_normalized(_full(x0))  # (n_full, n_full)
                H0 = H0_full[np.ix_(to_fit_indices, to_fit_indices)]    # (d, d)
                H0 += self._hessian_reg * np.eye(d)                           # ridge for stability (Tikhonov regularization)
                eigvals = np.linalg.eigvalsh(H0)
                if np.all(eigvals > 0):
                    _L     = np.linalg.cholesky(H0)   # p ~ N(0, M) via L @ z
                    _M_inv = np.linalg.inv(H0)         # position update uses M_inv @ p
                    _mass_type = "symbolic"
                    print(f"  [mass matrix] using symbolic Hessian (min eigval={eigvals.min():.4g}, reg={self._hessian_reg})")
                else:
                    warnings.warn("HMCFitter: Hessian mass matrix not PD; using identity.")
            except Exception as _he:
                warnings.warn(f"HMCFitter: mass matrix failed ({_he}); using identity.")

        def _KE(p):
            """Kinetic energy: 0.5 * p^T M^{-1} p (identity if _M_inv is None)."""
            if _M_inv is None:
                return 0.5 * float(np.dot(p, p))
            return 0.5 * float(p @ _M_inv @ p)

        def _sample_p():
            """Sample momentum from N(0, M) (standard normal if _L is None)."""
            z = rng.standard_normal(d)
            return z if _L is None else _L @ z

        # ---- checkpoint path ------------------------------------------
        _checkpoint_path = None
        if self._output_dir is not None:
            _checkpoint_path = os.path.join(self._output_dir, "hmc_checkpoint.npz")

        def _save_checkpoint(samples_so_far, q, lp, step, n_acc):
            if _checkpoint_path is None:
                return
            np.savez(_checkpoint_path,
                     samples=samples_so_far,
                     q_current=q,
                     lp_current=np.array(lp),
                     step_size=np.array(step),
                     n_accepted=np.array(n_acc),
                     rng_state=np.array(rng.bit_generator.state, dtype=object))
            print(f"  [checkpoint] {len(samples_so_far)} samples saved -> {_checkpoint_path}")

        # ---- resume from checkpoint ------------------------------------
        _resuming       = False
        _resume_samples = None   # samples already collected in a prior run
        q_current       = x0.copy()
        lp_current, _   = logprob_and_grad(q_current)
        n_accepted       = 0

        if self._resume_from is not None:
            _rpath = (os.path.join(self._resume_from, "hmc_checkpoint.npz")
                      if os.path.isdir(self._resume_from) else self._resume_from)
            if os.path.exists(_rpath):
                print(f"HMCFitter: resuming from {_rpath}")
                _ckpt           = np.load(_rpath, allow_pickle=True)
                _resume_samples = _ckpt['samples']                       # (k, d)
                q_current       = _ckpt['q_current'].copy()
                self._step_size = float(_ckpt['step_size'])
                lp_current, _   = logprob_and_grad(q_current)
                n_accepted      = int(_ckpt['n_accepted'])
                rng.bit_generator.state = _ckpt['rng_state'].item()
                _resuming = True
                _n_have   = len(_resume_samples)
                print(f"  [resume] {_n_have} samples already collected, "
                      f"need {max(0, self._n_samples - _n_have)} more, "
                      f"adapted step_size={self._step_size:.5g}")
            else:
                warnings.warn(f"HMCFitter: checkpoint '{_rpath}' not found; starting fresh.")

        _n_remaining = max(0, self._n_samples - (len(_resume_samples) if _resume_samples is not None else 0))

        def grad_fn(q):
            _, g = logprob_and_grad(q)
            return g

        n_width = len(str(self._n_samples))

        # ---- pre-sampling diagnostics (skipped on resume) --------------
        if not _resuming:
            g0 = grad_fn(x0)
            print(f"  [diagnostic] gradient norm at x0:  {np.linalg.norm(g0):.6g}")
            print(f"  [diagnostic] gradient per param:   "
                  + ", ".join(f"{n}={v:.4g}" for n, v in zip(unfixed_names, g0)))
            _p_test = _sample_p()
            _H0 = -lp_current + _KE(_p_test)
            _q_p, _p_p = _leapfrog(x0, _p_test, grad_fn, self._step_size, self._n_steps, _M_inv)
            _lp_p, _ = logprob_and_grad(_q_p)
            _H1 = -_lp_p + _KE(_p_p)
            print(f"  [diagnostic] test proposal:  H_current={_H0:.6g}  H_prop={_H1:.6g}"
                  f"  delta_H={_H1 - _H0:.6g}  log_accept={_H0 - _H1:.6g}")
            # reset rng so diagnostics don't consume samples
            rng = np.random.default_rng(self._random_seed)

        print(f"HMCFitter: drawing {self._n_samples} samples total "
              f"({_n_remaining} remaining) "
              f"(step_size={self._step_size}, n_steps={self._n_steps}, "
              f"gradient={self._gradient_type}, mass={_mass_type}, "
              f"parameterization={nc_label}{_adapt_note if not _resuming else ''})",
              flush=True)
        fit_start   = time.time()
        raw_samples = np.empty((_n_remaining, d))
        # ----------------------------------------------------------------

        try:
            # ---- burn-in phase with dual averaging -------------------------
            if self._burn_in > 0 and not _resuming:
                print(f"  [burn-in: {self._burn_in} steps, "
                      f"dual averaging -> target_accept={self._target_accept:.2f}]")

                # Nesterov dual averaging hyperparameters (Stan defaults)
                _mu          = np.log(10.0 * self._step_size)  # shrinkage target
                _gamma       = 0.05   # adaptation regularization
                _t0          = 10.0   # stability offset
                _kappa       = 0.75   # Polyak-Ruppert decay exponent
                _H_bar       = 0.0                        # dual averaging statistic
                _log_eps_bar = np.log(self._step_size)    # running log step-size average
                _eps         = self._step_size            # current (instantaneous) step size

                bi_accepted = 0
                for i in range(self._burn_in):
                    m = i + 1  # 1-indexed
                    p_current = _sample_p()
                    H_current = -lp_current + _KE(p_current)
                    q_prop, p_prop = _leapfrog(q_current, p_current,
                                               grad_fn, _eps, self._n_steps, _M_inv)
                    lp_prop, _ = logprob_and_grad(q_prop)
                    H_prop = -lp_prop + _KE(p_prop)

                    log_accept_ratio = H_current - H_prop
                    alpha = (min(1.0, np.exp(log_accept_ratio))
                             if np.isfinite(log_accept_ratio) else 0.0)

                    if np.log(rng.uniform()) < log_accept_ratio:
                        q_current  = q_prop
                        lp_current = lp_prop
                        bi_accepted += 1

                    # ---- dual averaging update ----------------------------
                    # H_bar tracks (target - alpha) with shrinking weight
                    w      = 1.0 / (m + _t0)
                    _H_bar = (1.0 - w) * _H_bar + w * (self._target_accept - alpha)
                    # Instantaneous step size
                    _log_eps     = _mu - (m ** 0.5 / _gamma) * _H_bar
                    _eps         = float(np.clip(np.exp(_log_eps), 1e-10, 1e3))
                    # Polyak-Ruppert running average (stabilises the frozen value)
                    _log_eps_bar = ((m ** -_kappa) * _log_eps
                                    + (1.0 - m ** -_kappa) * _log_eps_bar)

                    if (i + 1) % self._report_steps == 0 or (i + 1) == self._burn_in:
                        pct = 100 * (i + 1) / self._burn_in
                        print(f"    burn-in {i+1:>{len(str(self._burn_in))}d}/{self._burn_in} "
                              f"({pct:5.1f}%)  step_size={_eps:.5g}")

                # Freeze the Polyak-Ruppert average as the sampling step size
                self._step_size = float(np.clip(np.exp(_log_eps_bar), 1e-10, 1e3))
                print(f"  [burn-in done]  acceptance: {bi_accepted / self._burn_in:.2f}  "
                      f"adapted step_size: {self._step_size:.6g}  "
                      f"elapsed: {time.time() - fit_start:.1f}s")

            # ---- sampling phase ----------------------------------------
            _n_have_start = len(_resume_samples) if _resume_samples is not None else 0
            print(f"  [sampling: {_n_remaining} steps ({_n_have_start} -> {self._n_samples} total)]")
            for i in range(_n_remaining):
                p_current = _sample_p()
                H_current = -lp_current + _KE(p_current)
                q_prop, p_prop = _leapfrog(q_current, p_current,
                                           grad_fn, self._step_size, self._n_steps, _M_inv)
                lp_prop, _ = logprob_and_grad(q_prop)
                H_prop = -lp_prop + _KE(p_prop)
                if np.log(rng.uniform()) < H_current - H_prop:
                    q_current  = q_prop
                    lp_current = lp_prop
                    n_accepted += 1
                raw_samples[i] = q_current
                total_so_far = _n_have_start + i + 1
                if (i + 1) % self._report_steps == 0 or (i + 1) == _n_remaining:
                    elapsed     = time.time() - fit_start
                    accept_rate = n_accepted / total_so_far
                    pct_done    = 100 * total_so_far / self._n_samples
                    print(f"    {total_so_far:>{n_width}d}/{self._n_samples} samples "
                          f"({pct_done:5.1f}%)  "
                          f"acceptance: {accept_rate:.2f}  "
                          f"elapsed: {elapsed:.1f}s")
                # periodic checkpoint
                if (self._checkpoint_steps > 0
                        and (i + 1) % self._checkpoint_steps == 0
                        and _checkpoint_path is not None):
                    _all_so_far = (np.vstack([_resume_samples, raw_samples[:i+1]])
                                   if _resume_samples is not None else raw_samples[:i+1])
                    _save_checkpoint(_all_so_far, q_current, lp_current,
                                     self._step_size, n_accepted)

            self._success = True

        except KeyboardInterrupt:
            # Save whatever we have before exiting
            _collected = i + 1 if 'i' in dir() else 0
            if _collected > 0 and _checkpoint_path is not None:
                _partial = (np.vstack([_resume_samples, raw_samples[:_collected]])
                            if _resume_samples is not None else raw_samples[:_collected])
                _save_checkpoint(_partial, q_current, lp_current,
                                 self._step_size, n_accepted)
                print(f"HMCFitter: interrupted after {len(_partial)} total samples. "
                      f"Resume with resume_from='{self._output_dir}'.")
            else:
                print("HMCFitter: sampling interrupted by user.")
            raw_samples = None
            self._success = False
        except Exception as exc:
            warnings.warn(f"HMCFitter: HMC raised an exception: {exc}")
            raw_samples = None
            n_accepted  = 0
            self._success = False

        total_time = time.time() - fit_start

        # Combine resume samples (already post-burn-in) with new raw_samples.
        # burn_in slice only applies to a fresh run (not resume).
        if raw_samples is not None and raw_samples.shape[0] > 0:
            if _resume_samples is not None:
                # resume: all resume_samples + all new samples are post-burn-in
                _all_raw = np.vstack([_resume_samples, raw_samples])
                # save final checkpoint (in z-space, before back-transform)
                _save_checkpoint(_all_raw, q_current, lp_current,
                                 self._step_size, n_accepted)
                self._samples = _all_raw.copy()
            else:
                self._samples = raw_samples[self._burn_in:].copy()
                # Save a final checkpoint so a completed run can be resumed
                # (extended) in exactly the same way as an interrupted one.
                _save_checkpoint(self._samples, q_current, lp_current,
                                 self._step_size, n_accepted)

            # Transform samples from z-space back to theta-space
            if np.any(nc_mask):
                self._samples = self._samples.copy()
                self._samples[:, nc_mask] = (prior_means[nc_mask]
                                             + self._samples[:, nc_mask]
                                             * prior_stds[nc_mask])
            if _resume_samples is not None:
                self._acceptance_rate = n_accepted / self._samples.shape[0]
            else:
                self._acceptance_rate = n_accepted / raw_samples.shape[0]
            print(f"HMCFitter: done.  "
                  f"{self._samples.shape[0]} post-burn-in samples  |  "
                  f"acceptance rate: {self._acceptance_rate:.2f}  |  "
                  f"{total_time:.1f} s")
            if self._acceptance_rate < 0.1:
                warnings.warn(
                    f"HMCFitter: acceptance rate is very low "
                    f"({self._acceptance_rate:.2f}).  "
                    "Try reducing step_size."
                )
            elif self._acceptance_rate > 0.95:
                warnings.warn(
                    f"HMCFitter: acceptance rate is very high "
                    f"({self._acceptance_rate:.2f}).  "
                    "Try increasing step_size or n_steps for better mixing."
                )
        else:
            self._samples         = None
            self._acceptance_rate = np.nan
            self._success         = False
            warnings.warn("HMCFitter: no samples were collected.")

        # Store metadata
        self._total_time    = total_time
        self._unfixed_names = unfixed_names
        self._to_fit_mask   = to_fit_mask
        self._to_fit_indices = to_fit_indices

        self._update_fit_df()

        if fit_summary_file:
            self._save_fit_summary(fit_summary_file)
        if fit_results_file and self._fit_df is not None:
            self._fit_df.to_csv(fit_results_file, index=False)
            print(f"HMCFitter: fit results saved to {fit_results_file}")

    # ------------------------------------------------------------------
    # _update_fit_df  (mirrors EmceeFitter strategy)
    # ------------------------------------------------------------------

    def _update_fit_df(self):
        """Populate self._fit_df from HMC samples."""
        param_df     = self.param_df
        fixed_mask   = np.array(param_df["fixed"], dtype=bool)
        unfixed_mask = ~fixed_mask

        # Sync param_df metadata into _fit_df so displayed values are current
        for col in ["guess", "fixed", "lower_bound", "upper_bound",
                    "prior_mean", "prior_std"]:
            if col in param_df.columns:
                self._fit_df[col] = param_df[col].values

        # Ensure all required result columns are present
        for col in ["estimate", "std", "low_95", "high_95"]:
            if col not in self._fit_df.columns:
                self._fit_df[col] = np.nan

        # Fixed params: estimate = guess
        self._fit_df.loc[fixed_mask, "estimate"] = np.array(
            param_df.loc[fixed_mask, "guess"], dtype=float
        )

        s = self._samples
        if s is None or s.shape[0] == 0:
            return

        estimate = get_kde_max(s)
        std      = np.std(s, axis=0)

        n         = s.shape[0]
        lo_idx    = max(0,   int(round(0.025 * n)))
        hi_idx    = min(n-1, int(round(0.975 * n)))
        s_sorted  = np.sort(s, axis=0)          # single allocation: (n_samples, n_params)
        low_95    = s_sorted[lo_idx]
        high_95   = s_sorted[hi_idx]

        self._fit_df.loc[unfixed_mask, "estimate"] = estimate
        self._fit_df.loc[unfixed_mask, "std"]      = std
        self._fit_df.loc[unfixed_mask, "low_95"]   = low_95
        self._fit_df.loc[unfixed_mask, "high_95"]  = high_95

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def fit_df(self):
        """Fit results DataFrame."""
        return self._fit_df

    @property
    def samples(self):
        """
        Post-burn-in HMC samples. Shape: (n_post_burnin, n_unfixed_params).
        None if no fit or sampling failed.
        """
        return self._samples

    @property
    def success(self):
        """Whether sampling completed without error."""
        return self._success

    @property
    def fit_info(self):
        """Summary dictionary of HMC run configuration and results."""
        info = {
            "Backend":              "HMC (leapfrog, built-in)",
            "Gradient type":        getattr(self, "_gradient_type", "unknown"),
            "Parameterization":     "non-centered" if getattr(self, "_non_centered", False) else "centered",
            "n_samples":            self._n_samples,
            "burn_in":              self._burn_in,
            "initial_step_size":    getattr(self, "_initial_step_size", self._step_size),
            "adapted_step_size":    self._step_size,
            "target_accept":        getattr(self, "_target_accept", 0.65),
            "n_steps_per_sample":   self._n_steps,
            "success":              self._success,
        }
        if hasattr(self, "_acceptance_rate"):
            info["acceptance_rate"] = self._acceptance_rate
        if self._samples is not None:
            info["post_burnin_samples"] = self._samples.shape[0]
        if hasattr(self, "_total_time"):
            info["total_time_seconds"] = self._total_time
        return info

    # ------------------------------------------------------------------
    # Helper: save fit summary CSV
    # ------------------------------------------------------------------

    def _save_fit_summary(self, path):
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["metric", "value"])
            writer.writerow(["total_time_seconds",   f"{self._total_time:.6f}"])
            writer.writerow(["gradient_type",         self._gradient_type])
            writer.writerow(["parameterization",      "non-centered" if getattr(self, "_non_centered", False) else "centered"])
            writer.writerow(["n_samples_requested",   self._n_samples])
            writer.writerow(["burn_in",                self._burn_in])
            writer.writerow(["initial_step_size",      getattr(self, "_initial_step_size", self._step_size)])
            writer.writerow(["adapted_step_size",      self._step_size])
            writer.writerow(["target_accept",          getattr(self, "_target_accept", 0.65)])
            writer.writerow(["n_steps_per_sample",     self._n_steps])
            writer.writerow(["success",                str(self._success)])
            if hasattr(self, "_acceptance_rate"):
                writer.writerow(["acceptance_rate", f"{self._acceptance_rate:.4f}"])
            if self._samples is not None:
                writer.writerow(["post_burnin_samples", self._samples.shape[0]])
        print(f"HMCFitter: fit summary saved to {path}")

    # ------------------------------------------------------------------
    # __repr__
    # ------------------------------------------------------------------

    def __repr__(self):
        out = ["HMCFitter\n---------\n"]
        out.append(f"fit has been run: {self._fit_has_been_run}\n")
        if self._fit_has_been_run:
            status = "converged" if self._success else "failed or interrupted"
            out.append(f"fit status:      {status}\n")
            out.append(f"gradient type:   {getattr(self, '_gradient_type', 'unknown')}\n")
            if hasattr(self, "_acceptance_rate") and np.isfinite(self._acceptance_rate):
                out.append(f"acceptance rate: {self._acceptance_rate:.2f}\n")
            if self._samples is not None:
                out.append(f"post-burn-in samples: {self._samples.shape[0]}\n")
            if self._fit_df is not None:
                out.append("\nfit results:\n")
                for line in repr(self._fit_df).split("\n"):
                    out.append(f"  {line}\n")
        return "".join(out)
