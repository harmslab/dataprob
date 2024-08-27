========
dataprob
========

.. image:: tests-badge
    :target: docs/badges/tests-badge.svg
.. image:: coverage-badge
    :target: docs/badges/coverage-badge.svg


dataprob was designed to allow experimentalists to fit parameters from arbitrary
models to experimental data. 

+ **ease of use:** Users write a python function that describes their model, 
  then load in their experimental data as a dataframe. A full analysis can
  be run with two python commands. 
+ **dataframe centric:** Users use a dataframe to specify parameter bounds,
  guesses, fixedness, and priors. Observed data can be passed in as a
  dataframe or numpy vector. All outputs are simple pandas dataframes. 
+ **consistent experience:** Users can run maximum-likelihood, bootstrap 
  resampling, or Bayesian MCMC analyses with an identical interface and nearly
  identical diagnostic outputs. 
+ **interpretable:** Provides simple diagnostic plots and runs tests assessing
  fit results, flagging problems with residuals and co-varying parameters. 

Simple example
==============

The following code generates noisy linear data and uses dataprob to find 
the maximum likelihood estimate of its slope and intercept. 

.. code-block:: python
    
    import dataprob
    import numpy as np

    # Generate "experimental" linear data (slope = 5, intercept = 5.7) that has
    # random noise on each point. 
    x_array = np.linspace(0,10,25)
    noise = np.random.normal(loc=0,scale=0.5,size=x_array.shape)
    y_obs = 5*x_array + 5.7 + noise

    # 1. Define a linear model
    def linear_model(m=1,b=1,x=[]):
        return m*x + b

    # 2. Set up the analysis. 'method' can be "ml", "mcmc", or "bootstrap"
    f = dataprob.setup(linear_model,
                       method="ml",
                       non_fit_kwargs={"x":x_array})

    # 3. Fit the parameters of linear_model model to y_obs, assuming uncertainty
    #    of 0.5 on each observed point. 
    f.fit(y_obs=y_obs,
          y_std=0.5)

    # 4. Access results
    print(f.fit_df)
    fig = dataprob.plot_summary(f)
    fig = dataprob.plot_corner(f)

The ``f.fit_df`` dataframe will look something like:

+-------+-------+----------+-------+--------+---------+-------+-----------+
| index | name  | estimate | std   | low_95 | high_95 | ...   | prior_std |
+=======+=======+==========+=======+========+=========+=======+===========+
| ``m`` | ``m`` | 5.009    | 0.045 | 4.817  | 5.202   | ...   | ``NaN``   |  
+-------+-------+----------+-------+--------+---------+-------+-----------+
| ``b`` | ``b`` | 5.644    | 0.274 |  4.465 | 6.822   | ...   | ``NaN``   |
+-------+-------+----------+-------+--------+---------+-------+-----------+

The plots will be:

.. image:: plot-summary
    :target: docs/simple-example_plot-summary.svg


.. image:: plot-corner
    :target: docs/simple-example_plot-corner.svg


Installation
============

We recommend installing dataprob with pip:

.. code-block:: bash

    pip install dataprob

To install from source and run tests:

.. code-block:: bash

    git clone https://github.com/harmslab/dataprob.git
    cd dataprob
    pip install .

    # to run test-suite
    pytest --runslow

Examples
========

A good way to learn how to use the library is by working through examples. The
following notebooks are included in the `dataprob/examples/` directory. They are
self-contained demonstrations in which dataprob is used to analyze various
classes of experimental data. The links below launch each notebook in Google
colab:

+ `linear.ipynb <linear-example_>`_: fit a linear model to noisy data (2 parameter, linear)
+ `binding.ipynb <binding-example_>`_: a single-site binding interaction (2 parameter, sigmoidal curve)
+ `hill-model.ipynb <hill-model-example_>`_: cooperative ligand binding (3 parameter, sigmoidal curve)
+ `michaelis-menten.ipynb <michaelis-menten-example>`_: Michaelis-Menten model of enzyme kinetics (2 parameter, sigmoidal curve)
+ `lagged-exponential.ipynb <lagged-exponential-example>`_: bacterial growth curve with initial lag phase (3 parameter, exponential)
+ `multi-gaussian.ipynb <multi-gaussian-example>`_: two overlapping normal distributions (6 parameter, Gaussian)
+ `periodic.ipynb <periodic-example>`_: periodic data (3 parameter, sine) 
+ `polynomial.ipynb <polynomial-example>`_: nonlinear data with no obvious form (5 parameter, polynomial)
+ `linear-extrapolation-folding.ipynb <linear-extrapolation-folding-example>`_: protein equilibrium unfolding data (6 parameter, linear embedded in sigmoidal)

Documentation
=============

Full documentation is on `readthedocs <https://dataprob.readthedocs.io>`_.
