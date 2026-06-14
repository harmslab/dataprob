.. include:: ../links.rst

==========================
Hamiltonian Monte Carlo
==========================

**method key:** ``"hmc"``

When to use
===========

The HMC fitter samples the Bayesian posterior using a self-contained leapfrog
`Hamiltonian Monte Carlo <hmc-wiki_>`_ implementation (pure numpy — no
autodiff framework required). Use HMC when:

+ Your parameters are strongly correlated and ML fitting gives unreliable
  uncertainties.
+ You have an analytical Jacobian available (either from the linkage symbolic
  framework or a custom implementation), enabling efficient gradient-based
  sampling.
+ You want checkpoint/resume support for long runs.
+ You want to avoid the PyMC/PyTensor dependency while still getting
  gradient-based sampling.

Without a Jacobian, HMC falls back to a forward finite-difference gradient
and is considerably slower than with a symbolic one.

Algorithm
=========

HMC uses `leapfrog integration <leapfrog_>`_ to propose moves in the joint
space of parameters :math:`q` and auxiliary momenta :math:`p`:

.. math::

    p_{1/2} &= p_{0} + \tfrac{\varepsilon}{2} \nabla \ln\pi(q_{0}) \\
    q_{i+1} &= q_{i} + \varepsilon\, M^{-1} p_{i+1/2} \\
    p_{i+1/2} &= p_{i-1/2} + \varepsilon \nabla \ln\pi(q_{i+1}) \\
    p_{L}   &= p_{L-1/2} + \tfrac{\varepsilon}{2} \nabla \ln\pi(q_{L})

After :math:`L` leapfrog steps the proposal :math:`(q^{*}, p^{*})` is accepted
or rejected via a `Metropolis criterion <metropolis-crit_>`_.

**Burn-in and step-size adaptation** are handled by
`dual averaging <dual-averaging_>`_ (the same scheme used by Stan), which
automatically tunes :math:`\varepsilon` during the burn-in phase to hit the
target acceptance rate.

Gradient source
===============

The HMC fitter detects an analytical Jacobian via duck typing: if the bound
method's owning object exposes a callable ``jacobian_normalized(params)``
attribute, it is used to compute :math:`\nabla \ln\pi` analytically.

.. code-block:: text

    HMC status line example:
    gradient=symbolic   ← analytical Jacobian in use
    gradient=finite-difference  ← falling back to forward differences

When the symbolic Jacobian is available, the fitter calls the model's
``model_normalized(params)`` first (forward pass), then immediately calls
``jacobian_normalized(params)``. If the Jacobian implementation caches the
result of the most recent forward pass, the second call skips the forward
model evaluation — giving one model evaluation per leapfrog step instead of
two.

Mass matrix
===========

If the model object also exposes a callable ``hessian_normalized(params)``,
the HMC fitter uses the Hessian evaluated at the starting point as a diagonal
mass matrix :math:`M`. This adapts the momentum scale to the curvature of the
posterior, dramatically improving mixing when parameters have very different
scales.

.. code-block:: text

    HMC status line example:
    mass=symbolic   ← Hessian mass matrix in use
    mass=identity   ← standard isotropic mass matrix

If the Hessian is not positive-definite after Tikhonov regularisation, the
fitter warns and falls back to the identity mass matrix.

Checkpointing
=============

HMC supports save/resume for long runs. Set ``output_dir`` and
``checkpoint_steps`` to periodically save progress:

.. code-block:: python

    f.fit(y_obs=y_obs,
          y_std=y_std,
          n_samples=5000,
          output_dir="results/hmc",
          checkpoint_steps=100)

If the run is interrupted (``KeyboardInterrupt``), the current samples are
saved automatically. Resume by passing the same ``output_dir``:

.. code-block:: python

    f.fit(y_obs=y_obs,
          y_std=y_std,
          n_samples=5000,
          output_dir="results/hmc",
          resume_from="results/hmc")

Priors
======

Priors are set via ``param_df`` in the same way as the other Bayesian fitters:

+ ``prior_mean = NaN``, ``prior_std = NaN`` → uniform between bounds.
+ Both set → Gaussian prior :math:`\mathcal{N}(\text{prior\_mean},\
  \text{prior\_std}^{2})`.
+ Bounds enforced with a repelling gradient: proposals outside bounds are
  reflected back toward the interior.

Usage
=====

Minimal standalone use with a plain function:

.. code-block:: python

    import dataprob
    import numpy as np

    def linear(m, b, x):
        return m*x + b

    x = np.linspace(0, 10, 30)
    y_obs = 3*x + 1.5 + np.random.normal(0, 0.5, size=x.shape)

    f = dataprob.setup(linear,
                       method="hmc",
                       non_fit_kwargs={"x": x})

    f.param_df.loc["m", "lower_bound"] = 0.0
    f.param_df.loc["m", "upper_bound"] = 10.0
    f.param_df.loc["b", "lower_bound"] = -5.0
    f.param_df.loc["b", "upper_bound"] = 10.0
    f.param_df.loc["m", "guess"] = 3.0
    f.param_df.loc["b", "guess"] = 1.0

    f.fit(y_obs=y_obs,
          y_std=0.5,
          n_samples=2000,
          burn_in=400,
          step_size=0.05,
          n_steps=10,
          target_accept=0.65,
          output_dir="results/hmc",
          checkpoint_steps=100)

    print(f.fit_df[["estimate", "std", "low_95", "high_95"]])

With a symbolic Jacobian from the linkage ``GlobalModel``:

.. code-block:: python

    import linkage, dataprob

    gm = linkage.GlobalModel(model_name="GenericBindingModel",
                             model_spec=model_spec,
                             expt_list=all_expts,
                             use_symbolic_jacobian=True)

    f = dataprob.setup(gm.model_normalized,
                       method="hmc",
                       vector_first_arg=True,
                       fit_parameters=gm.parameter_names)

    # configure param_df ...

    f.fit(y_obs=gm.y_obs_normalized,
          y_std=gm.y_std_normalized,
          n_samples=2000,
          burn_in=250,
          output_dir="results/hmc",
          checkpoint_steps=100)

fit() kwargs
============

+---------------------+--------------------------------------------------+----------+
| Argument            | Effect                                           | Default  |
+=====================+==================================================+==========+
| ``n_samples``       | Total proposals to draw (before burn-in removal) | ``2000`` |
+---------------------+--------------------------------------------------+----------+
| ``burn_in``         | Leading samples to discard; dual averaging       | ``400``  |
|                     | tunes step_size during this phase.               |          |
+---------------------+--------------------------------------------------+----------+
| ``step_size``       | Initial leapfrog step size :math:`\varepsilon`   | ``0.1``  |
+---------------------+--------------------------------------------------+----------+
| ``n_steps``         | Leapfrog steps per proposal :math:`L`            | ``20``   |
+---------------------+--------------------------------------------------+----------+
| ``target_accept``   | Dual-averaging target acceptance rate            | ``0.65`` |
+---------------------+--------------------------------------------------+----------+
| ``report_steps``    | Progress print frequency                         | ``100``  |
+---------------------+--------------------------------------------------+----------+
| ``random_seed``     | RNG seed for reproducibility                     | ``None`` |
+---------------------+--------------------------------------------------+----------+
| ``output_dir``      | Directory to save CSVs and checkpoints           | ``None`` |
+---------------------+--------------------------------------------------+----------+
| ``checkpoint_steps``| Save checkpoint every N sampling steps           | ``0``    |
+---------------------+--------------------------------------------------+----------+
| ``resume_from``     | Path to checkpoint dir or ``.npz`` file          | ``None`` |
+---------------------+--------------------------------------------------+----------+
| ``hessian_reg``     | Tikhonov regularisation for mass matrix          | ``1e-4`` |
+---------------------+--------------------------------------------------+----------+
| ``non_centered``    | Apply non-centered parameterization for params   | ``False``|
|                     | with Gaussian priors (see below)                 |          |
+---------------------+--------------------------------------------------+----------+

Non-centered parameterization
==============================

When ``non_centered=True``, any parameter with both ``prior_mean`` and
``prior_std`` set is reparameterized before sampling.  Instead of drawing
``theta`` directly, the sampler draws a standard-normal offset:

.. math::

    z \sim \mathcal{N}(0, 1), \qquad \theta = \mu + z\,\sigma

where :math:`\mu` = ``prior_mean`` and :math:`\sigma` = ``prior_std``.

**When to use it:** the non-centered form is preferred when the likelihood is
weak relative to the prior — i.e., when there are few observations per
parameter group. In that regime the centered posterior develops a narrow funnel
geometry that HMC explores slowly. The non-centered form decouples the offset
:math:`z` from the scale :math:`\sigma`, giving the sampler a much flatter
geometry to traverse.

Conversely, when the likelihood is strong (many observations per parameter),
the centered form is often better: the posterior is already tight and the
extra transformation adds unnecessary complexity.

Parameters with only bounds and no Gaussian prior are unaffected by this
option.

Interpreting the diagnostic output
===================================

At startup, before burn-in, HMC prints several diagnostics:

.. code-block:: text

    [diagnostic] gradient norm at x0: 1.234e+02
    [diagnostic] gradient per param:  KI=12.3, k_high=45.6, ...
    [diagnostic] test proposal:  H_current=142.3  H_prop=144.1  delta_H=1.8  log_accept=-1.8

A ``log_accept`` near zero indicates the initial step size is appropriate. Very
large ``delta_H`` (e.g. > 10) suggests the step size is too large or the
starting point is far from a high-probability region.

During sampling:

.. code-block:: text

    200/2000 samples (10.0%)  acceptance: 0.65  elapsed: 12.3s

Acceptance rates below 0.1 suggest the step size is too large (reduce
``step_size``). Rates above 0.95 suggest it is too small (increase
``step_size`` or ``n_steps``).

Outputs
=======

+ ``f.fit_df["estimate"]``: mode of the posterior marginal distribution,
  estimated by a `Gaussian kernel density estimator <kde_>`_.
+ ``f.fit_df["std"]``: standard deviation of the marginal posterior.
+ ``f.fit_df["low_95"]``, ``f.fit_df["high_95"]``: 95%
  `credible interval <credible-interval_>`_ determined numerically.
+ ``f.samples``: array of shape ``(n_post_burnin, n_params)`` containing the
  posterior samples.
+ ``f._acceptance_rate``: overall Metropolis acceptance rate.
+ ``f._step_size``: final adapted step size after dual averaging.
+ If ``output_dir`` is set: ``fit_summary.csv``, ``fit_results.csv``, and
  ``hmc_checkpoint.npz`` (updated periodically if ``checkpoint_steps > 0``).
