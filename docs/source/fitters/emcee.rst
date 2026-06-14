.. include:: ../links.rst

=====================
Bayesian MCMC (emcee)
=====================

**method key:** ``"emcee"``

When to use
===========

The emcee fitter samples the Bayesian posterior distribution of model
parameters using an affine-invariant ensemble sampler. Use emcee when:

+ You want the full posterior distribution of parameters, including all
  correlations.
+ Your model is moderately fast (the sampler calls the model thousands of
  times).
+ You do not have an analytical Jacobian available.

For models with analytical Jacobians and strongly correlated parameters,
the :doc:`hmc` fitter is generally faster and more efficient. For very complex
models or when you want NUTS-style sampling with autodiff, see :doc:`pymc`.

Algorithm
=========

emcee uses the `EnsembleSampler <emcee-ensemble-sampler_>`_, an
affine-invariant `MCMC <mcmc_>`_ algorithm that runs many parallel walkers
simultaneously. Each walker proposes a move based on the current positions of
the other walkers (the "stretch move" by default). A
`Metropolis criterion <metropolis-crit_>`_ decides whether to accept or reject
each proposal.

The posterior probability for a parameter vector :math:`\vec{x}` is:

.. math::

    \ln P(\vec{x} | y_{obs}) = \ln \mathcal{L}(\vec{x}) + \ln P(\vec{x})

where the log-likelihood is:

.. math::

    \ln \mathcal{L}(\vec{x}) = -\frac{1}{2} \sum_{i} \left[
        \frac{(y_{calc,i}(\vec{x}) - y_{obs,i})^{2}}{y_{std,i}^{2}}
        + \ln(2\pi\, y_{std,i}^{2})
    \right]

and :math:`\ln P(\vec{x})` is the log prior.

Priors
======

Each parameter's prior is controlled by its ``param_df`` entries:

+ ``prior_mean = NaN``, ``prior_std = NaN`` → uniform between bounds.
+ ``prior_mean`` and ``prior_std`` both set → Gaussian prior
  :math:`\mathcal{N}(\text{prior\_mean},\ \text{prior\_std}^{2})`.
+ Bounds + Gaussian prior together → truncated Gaussian, renormalized within
  bounds.

.. note::

    We strongly recommend setting at least bounds on every parameter. Without
    bounds, walkers can wander into regions of parameter space where the model
    is undefined or numerically unstable, which causes the sampler to stall.

Convergence
===========

emcee assesses convergence via the integrated
`autocorrelation time <emcee-autocorr_>`_ :math:`\tau` for each parameter.
The built-in convergence criterion is :math:`50\tau`. If this is not met,
dataprob issues a warning and ``f.fit_quality["success"]`` is ``False``.

To run until convergence automatically, set ``max_convergence_cycles > 1``:

.. code-block:: python

    f.fit(y_obs=y_obs, y_std=y_std, max_convergence_cycles=10)

Usage
=====

.. code-block:: python

    import dataprob
    import numpy as np

    def linear(m, b, x):
        return m*x + b

    x = np.linspace(0, 10, 30)
    y_obs = 3*x + 1.5 + np.random.normal(0, 0.5, size=x.shape)

    f = dataprob.setup(linear,
                       method="emcee",
                       non_fit_kwargs={"x": x})

    f.param_df.loc["m", "lower_bound"] = 0.0
    f.param_df.loc["m", "upper_bound"] = 10.0
    f.param_df.loc["b", "lower_bound"] = -5.0
    f.param_df.loc["b", "upper_bound"] = 10.0

    f.fit(y_obs=y_obs,
          y_std=0.5,
          num_walkers=50,
          num_steps=500,
          burn_in=0.2)

    print(f.fit_df[["estimate", "std", "low_95", "high_95"]])

fit() kwargs
============

+-------------------------+--------------------------------------------------+-----------+
| Argument                | Effect                                           | Default   |
+=========================+==================================================+===========+
| ``num_walkers``         | Number of parallel MCMC chains                   | ``100``   |
+-------------------------+--------------------------------------------------+-----------+
| ``use_ml_guess``        | Seed walker positions from an ML fit             | ``True``  |
+-------------------------+--------------------------------------------------+-----------+
| ``num_steps``           | Steps per walker                                 | ``100``   |
+-------------------------+--------------------------------------------------+-----------+
| ``burn_in``             | Fraction of steps to discard as burn-in          | ``0.1``   |
+-------------------------+--------------------------------------------------+-----------+
| ``max_convergence_cycles`` | Repeat until :math:`50\tau` criterion met     | ``1``     |
+-------------------------+--------------------------------------------------+-----------+
| ``output_dir``          | Directory to save ``fit_results.csv`` and        | ``None``  |
|                         | ``samples.csv``                                  |           |
+-------------------------+--------------------------------------------------+-----------+

Any additional keyword arguments are forwarded to
``emcee.EnsembleSampler.__init__``, allowing control over the move strategy
and other sampler internals.

Outputs
=======

+ ``f.fit_df["estimate"]``: mode of the posterior marginal distribution,
  estimated by a `Gaussian kernel density estimator <kde_>`_.
+ ``f.fit_df["std"]``: standard deviation of the marginal posterior.
+ ``f.fit_df["low_95"]``, ``f.fit_df["high_95"]``: 95%
  `credible interval <credible-interval_>`_ determined numerically.
+ ``f.samples``: array of shape ``(n_post_burnin_steps × n_walkers, n_params)``
  containing the posterior samples.
+ ``f.fit_quality``: includes autocorrelation convergence diagnostics.
