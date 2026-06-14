.. include:: ../links.rst

==================
Maximum Likelihood
==================

**method key:** ``"ml"``

When to use
===========

Maximum likelihood (ML) is the right starting point for most fitting problems.
It is fast, has no sampling overhead, and gives a point estimate with an
associated uncertainty. Use ML when:

+ Your parameters are not strongly correlated with one another.
+ You want a quick result to validate your model before running a slower
  Bayesian analysis.
+ You need a good starting guess to seed an HMC or emcee run.

If your parameters are strongly correlated — i.e. the data cannot
independently constrain them — ML will still find a solution, but the
reported uncertainties will be unreliable. In that case use HMC or emcee.

Algorithm
=========

ML minimises the weighted residual sum of squares using
`scipy.optimize.least_squares <scipy-least-squares_>`_:

.. math::

    \chi^{2}(\vec{x}) = \sum_{i} \left( \frac{y_{calc,i}(\vec{x}) - y_{obs,i}}{y_{std,i}} \right)^{2}

The default solver is the Trust Region Reflective (``"trf"``) algorithm, which
handles bounds natively. Any keyword argument accepted by
``scipy.optimize.least_squares`` can be passed through ``f.fit()``.

Symbolic Jacobian
=================

If the model object exposes a ``jacobian_normalized(params)`` method, the ML
fitter uses it as the analytic Jacobian for the optimiser instead of relying
on finite differences. This speeds up convergence and improves accuracy for
complex models. For most standalone functions this is not needed — scipy's
built-in finite-difference Jacobian is sufficient.

Parameter uncertainty
=====================

After optimisation, the ML fitter estimates parameter uncertainty from the
local curvature of the likelihood surface using the Gauss-Newton approximation
to the Hessian:

.. math::

    H \approx 2 J^{T} J

where :math:`J` is the Jacobian matrix returned by
``scipy.optimize.least_squares``. The covariance matrix is:

.. math::

    C = \frac{\chi^{2}_{red}}{1} \cdot (J^{T} J)^{-1}

Parameter uncertainties are the square roots of the diagonal of :math:`C`;
95% confidence intervals are computed from a t-distribution. Samples
(``f.samples``) are drawn from the multivariate normal
:math:`\mathcal{N}(\hat{\mu}, C)`.

.. note::

    This approach assumes that uncertainties are normally distributed around
    the ML estimate. It works well when parameters are well-constrained; it
    can underestimate uncertainty when parameters are strongly correlated.
    For correlated parameters, use the HMC or emcee fitters.

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
                       method="ml",
                       non_fit_kwargs={"x": x})

    f.param_df.loc["m", "guess"] = 3.0
    f.param_df.loc["b", "guess"] = 1.0

    f.fit(y_obs=y_obs, y_std=0.5)

    print(f.fit_df[["estimate", "std", "low_95", "high_95"]])

fit() kwargs
============

All keyword arguments are forwarded to ``scipy.optimize.least_squares``. Useful
ones include:

+----------------+--------------------------------------------+-------------------+
| Argument       | Effect                                     | Default           |
+================+============================================+===================+
| ``method``     | Solver: ``"trf"``, ``"dogbox"``, ``"lm"``  | ``"trf"``         |
+----------------+--------------------------------------------+-------------------+
| ``ftol``       | Convergence tolerance on cost function     | ``1e-8``          |
+----------------+--------------------------------------------+-------------------+
| ``xtol``       | Convergence tolerance on parameters        | ``1e-8``          |
+----------------+--------------------------------------------+-------------------+
| ``gtol``       | Convergence tolerance on gradient          | ``1e-8``          |
+----------------+--------------------------------------------+-------------------+
| ``x_scale``    | Parameter scaling; ``"jac"`` is adaptive   | ``1.0``           |
+----------------+--------------------------------------------+-------------------+
| ``max_nfev``   | Maximum function evaluations               | ``None``          |
+----------------+--------------------------------------------+-------------------+
| ``loss``       | Loss function: ``"linear"``, ``"huber"``   | ``"linear"``      |
+----------------+--------------------------------------------+-------------------+
| ``verbose``    | Verbosity: ``0``, ``1``, ``2``             | ``0``             |
+----------------+--------------------------------------------+-------------------+
| ``output_dir`` | Directory to save fit CSVs                 | ``None``          |
+----------------+--------------------------------------------+-------------------+
| ``num_samples``| Number of samples drawn from covariance    | ``100000``        |
+----------------+--------------------------------------------+-------------------+

Outputs
=======

+ ``f.fit_df["estimate"]``: maximum likelihood parameter estimate.
+ ``f.fit_df["std"]``: standard deviation from the inverse Hessian.
+ ``f.fit_df["low_95"]``, ``f.fit_df["high_95"]``: 95% confidence interval.
+ ``f.samples``: array of shape ``(num_samples, n_params)`` drawn from
  the covariance matrix.
+ ``f.fit_quality``: chi², residual tests, and convergence status.
+ If ``output_dir`` is set: ``fit_summary.csv`` and ``fit_results.csv``.
