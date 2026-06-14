.. include:: ../links.rst

====================
Bayesian MCMC (PyMC)
====================

**method key:** ``"pymc"``

When to use
===========

The PyMC fitter samples the Bayesian posterior using PyMC's NUTS (No-U-Turn
Sampler) with multiple parallel chains. Use PyMC when:

+ You want NUTS-style sampling with automatic step-size tuning and
  multiple chains for convergence diagnostics (R-hat, ESS).
+ You have an analytical Jacobian available (enables NUTS; without one,
  the fitter falls back to a slower Slice sampler).
+ You want the arviz diagnostic outputs (R-hat, effective sample size).

If you have a symbolic Jacobian from linkage or another source but want a
simpler interface without the PyMC/PyTensor dependency, use :doc:`hmc`.

.. note::

    PyMC is not installed by default. Install it separately::

        pip install pymc

Algorithm
=========

PyMC's NUTS sampler is a gradient-based MCMC method that avoids the random
walk behaviour of simpler samplers by using the gradient of the log-posterior
to propose efficient moves. It automatically tunes the step size and
trajectory length during warm-up.

When a Jacobian is available, the fitter builds a custom `PyTensor Op
<pytensor_>`_ that computes the model output and its Jacobian together in a
single pass. PyTensor's autodiff graph then uses the Jacobian for the
gradient of the log-posterior without a second forward model evaluation.

Without a Jacobian, the fitter falls back to PyMC's gradient-free
Slice sampler with a warning.

Symbolic Jacobian
=================

The PyMC fitter detects a Jacobian via duck typing: if the bound method's
owning object exposes a callable ``jacobian_normalized(params)`` attribute,
it is used.

.. important::

    The ``jacobian_normalized`` implementation should cache the result of
    the most recent forward model evaluation and reuse it when called with the
    same parameter vector. This avoids a redundant model evaluation inside
    each gradient step. If caching is not implemented the result will be
    correct but the forward pass will be computed twice per gradient call.

Priors
======

Priors are set via ``param_df`` in the same way as the :doc:`emcee` fitter:

+ ``prior_mean = NaN``, ``prior_std = NaN`` → ``pm.Uniform(lower, upper)``
  (finite bounds required).
+ Both ``prior_mean`` and ``prior_std`` set →
  ``pm.TruncatedNormal(mu, sigma, lower, upper)``.

.. note::

    PyMC requires **finite bounds** for all parameters using uniform priors.
    An error is raised at setup time if a parameter has ``-inf`` or ``inf``
    bounds and no Gaussian prior.

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
                       method="pymc",
                       non_fit_kwargs={"x": x})

    # Finite bounds are required for Uniform priors
    f.param_df.loc["m", "lower_bound"] = 0.0
    f.param_df.loc["m", "upper_bound"] = 10.0
    f.param_df.loc["b", "lower_bound"] = -5.0
    f.param_df.loc["b", "upper_bound"] = 10.0

    f.fit(y_obs=y_obs,
          y_std=0.5,
          draws=1000,
          tune=1000,
          chains=4,
          output_dir="results/pymc")

    print(f.fit_df[["estimate", "std", "low_95", "high_95"]])

fit() kwargs
============

+------------------+--------------------------------------------------+-----------+
| Argument         | Effect                                           | Default   |
+==================+==================================================+===========+
| ``draws``        | Posterior samples per chain                      | ``1000``  |
+------------------+--------------------------------------------------+-----------+
| ``tune``         | Warm-up / tuning steps per chain                 | ``1000``  |
+------------------+--------------------------------------------------+-----------+
| ``chains``       | Number of independent chains                     | ``4``     |
+------------------+--------------------------------------------------+-----------+
| ``target_accept``| NUTS target acceptance rate (NUTS only)          | ``0.9``   |
+------------------+--------------------------------------------------+-----------+
| ``output_dir``   | Directory to save CSVs and diagnostics           | ``None``  |
+------------------+--------------------------------------------------+-----------+

Any additional keyword arguments are forwarded to ``pm.sample()``.

Outputs
=======

+ ``f.fit_df["estimate"]``: mode of the posterior marginal distribution.
+ ``f.fit_df["std"]``: standard deviation of the marginal posterior.
+ ``f.fit_df["low_95"]``, ``f.fit_df["high_95"]``: 95%
  `credible interval <credible-interval_>`_.
+ ``f.samples``: array of shape ``(draws × chains, n_params)``.
+ ``f._fit_result``: the raw ``arviz.InferenceData`` object for advanced
  diagnostics.
+ If ``output_dir`` is set: ``fit_results.csv``, ``samples.csv`` (thinned to
  10 000 rows), and ``fit_summary.csv`` (with R-hat and ESS if arviz is
  installed).
