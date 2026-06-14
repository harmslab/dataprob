.. include:: ../links.rst

=========
Bootstrap
=========

**method key:** ``"bootstrap"``

When to use
===========

Bootstrap resampling provides parameter uncertainty estimates that do not
assume normally distributed posteriors. Use bootstrap when:

+ You suspect your parameter posterior distributions are non-Gaussian (e.g.
  asymmetric or multi-modal).
+ You want an uncertainty estimate that makes no assumptions about the
  likelihood surface shape.
+ Your model is fast enough to run many (100+) times in reasonable time.

For problems where parameters are strongly correlated, bootstrap will reveal
the correlation structure but may be slow. HMC is generally more efficient
in those cases.

Algorithm
=========

The bootstrap method runs maximum likelihood estimation on a large number of
pseudo-replicate datasets. For each replicate:

1. A new ``y_obs`` is sampled by drawing from
   :math:`\mathcal{N}(y_{obs,i},\ y_{std,i})` independently for each
   observation.
2. ``scipy.optimize.least_squares`` finds the maximum likelihood parameters
   for that replicate using the *unweighted* residuals (because uncertainty is
   already captured by the sampling):

   .. math::

       r_{i}(\vec{x}) = y_{calc,i}(\vec{x}) - y_{obs,i}

3. The parameter vector from each successful fit is stored as a sample.

The resulting collection of parameter vectors approximates the conditional
parameter distribution given the data and the model.

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
                       method="bootstrap",
                       non_fit_kwargs={"x": x})

    f.fit(y_obs=y_obs,
          y_std=0.5,
          num_bootstrap=500)

    print(f.fit_df[["estimate", "std", "low_95", "high_95"]])

fit() kwargs
============

+------------------+-----------------------------------------------+----------+
| Argument         | Effect                                        | Default  |
+==================+===============================================+==========+
| ``num_bootstrap``| Number of pseudo-replicate fits to run        | ``100``  |
+------------------+-----------------------------------------------+----------+
| ``output_dir``   | Directory to save ``fit_results.csv`` and     | ``None`` |
|                  | ``samples.csv``                               |          |
+------------------+-----------------------------------------------+----------+

All remaining keyword arguments are forwarded to ``scipy.optimize.least_squares``
(same options as the :doc:`ml` fitter).

Outputs
=======

+ ``f.fit_df["estimate"]``: mode of the parameter distributions across
  pseudo-replicates, estimated by a
  `Gaussian kernel density estimator <kde_>`_.
+ ``f.fit_df["std"]``: standard deviation of the marginal distribution.
+ ``f.fit_df["low_95"]``, ``f.fit_df["high_95"]``: 95% interval determined
  numerically from sorted samples — no normality assumption.
+ ``f.samples``: array of shape ``(num_bootstrap, n_params)`` containing all
  pseudo-replicate parameter estimates. Rows from failed fits contain
  ``np.nan``.
+ ``f.fit_quality``: includes the number of successful and failed replicates.

.. note::

    If some bootstrap replicates fail to converge, a warning is issued and
    the failed rows in ``f.samples`` contain ``nan``. Statistics are computed
    on the successful rows only.
