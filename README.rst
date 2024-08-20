========
dataprob
========

.. image:: docs/badges/coverage-badge.svg

Library for using likelihoods (the probability of observed data given a model)
to extract parameter estimates for models describing experimental data. Can do
maximum likelihood, Bayesian Markov-Chain Monte Carlo sampling, and bootstrap
sampling using a simple, consistent API.  

The docs are in progress. See the "examples" directory for jupyter notebooks 
demonstrating the library. 

Basic workflow
==============

The following code block generates some noisy linear data and does a maximum
likelihood fit to estimate the slope and intercept. 

.. code-block:: python
    
    import dataprob
    import numpy as np

    # Define a linear model
    def linear_model(m=1,b=1,x=[]): return m*x + b

    # Generate some data with this model (m = 5, b = 5.7), adding
    # random noise with a standard deviation of 0.5 to each point
    x_array = np.linspace(0,10,25)
    noise = np.random.normal(loc=0,scale=0.5,size=x_array.shape[0])
    y_obs = linear_model(5,5.7,x_array) + noise
    
    # set up analysis. "ml", "mcmc" or "bootstrap" allowed
    f = dataprob.setup(linear_model,
                       method="ml",
                       non_fit_kwargs={"x":x_array})

    # Fit the wrapped model to y_obs, setting our estimated uncertainty
    # on each observed point to 0.5
    f.fit(y_obs=y_obs,
          y_std=0.5)

The ``f.fit_df`` dataframe will look something like:

+-------+-------+----------+-------+--------+---------+-------+-----------+
| index | name  | estimate | std   | low_95 | high_95 | ...   | prior_std |
+=======+=======+==========+=======+========+=========+=======+===========+
| ``m`` | ``m`` | 5.009    | 0.045 | 4.817  | 5.202   | ...   | ``NaN``   |  
+-------+-------+----------+-------+--------+---------+-------+-----------+
| ``b`` | ``b`` | 5.644    | 0.274 |	4.465 | 6.822   | ...   | ``NaN``   |
+-------+-------+----------+-------+--------+---------+-------+-----------+


Fitters
=======

There are three different analyses possible:

+ *ml*: Do a maximum likelihood fit, regressing model parameters against
  observed data. 
+ *mcmc*: Use Markov-Chain Monte Carlo to estimate the posterior distributions
  for model parameters. 
+ *bootstrap*: Estimate parameter distributions consistent with observed data
  by sampling observation uncertainty and using maximum likelihood to estimate
  fit parameters on each pseudoreplicate dataset. 

.. _full-parameters-ref:

Parameters
==========

Parameters are defined in the `f.param_df` dataframe. It has the following 
columns:

+-----------------+---------------------------------------------------------+
| key             | value                                                   |
+=================+=========================================================+
| ``name``        | string name of the parameter. should not be changed     |
|                 | by the user once fitter is initialized.                 |
+-----------------+---------------------------------------------------------+
| ``guess``       | guess as single float value (must be non-nan and        |
|                 | within bounds if specified)                             |
+-----------------+---------------------------------------------------------+
| ``fixed`        | whether or not parameter can vary. ``True`` or ``False``|
+-----------------+---------------------------------------------------------+
| ``lower_bound`` | single float value; ``-np.inf`` allowed; ``None``,      |
|                 | ``np.nan``, or ``pd.NA`` interpreted as `np.inf`.       |
+-----------------+---------------------------------------------------------+
| ``upper_bound`` | single float value; ``-np.inf`` allowed; ``None``,      |
|                 | ``np.nan``, or ``pd.NA`` interpreted as ``np.inf``.     |
+-----------------+---------------------------------------------------------+
| ``prior_mean``  | single float value; ``np.nan`` allowed (see below)      |
+-----------------+---------------------------------------------------------+
| ``prior_std``   | single float value; ``np.nan`` allowed (see below)      |
+-----------------+---------------------------------------------------------+

Gaussian priors are specified using the ``prior_mean`` and ``prior_std`` columns, 
declaring the prior mean and standard deviation. If both are set to ``np.nan``
for a parameter, the prior for that parameter is set to uniform between the
parameter bounds. If either ``prior_mean`` or ``prior_std`` is set to a non-nan
value, both must be non-nan to define the prior. When set, ``prior_std`` must be
greater than zero. Neither can be ``np.inf``. Both a gaussian prior and bounds
may be specified. 

The ``name`` column is set when the dataframe is initialized. This defines
the names of the parameters, which cannot be changed later. The ``name``
column is used as the index for the dataframe, allowing commands like the 
following:

.. code-block:: python

    # set the guess of parameter m to 1.0.
    f.param_df["m","guess"] = 10.0

You can also edit the dataframe en masse and load in directly:

.. code-block:: python

    df = f.param_df.copy()

    # do lots of edits to dataframe
    # ... 
    # ...
    # then:

    f.param_df = df


Model definition
================

The software can regress float parameters to any function that returns
a numpy array the same length as ``y_obs``. The function can be a conventional
function, the method of a complicated class, or any other object with a 
``__call__`` attribute. 

The arguments passed to the wrapped function can be treated as either fittable
or non-fittable. Fittable parameters will be regressed; non-fittable parameters
are passed to the function as fixed values every time it is called. The software
uses the `function signature <https://docs.python.org/3/library/inspect.html#inspect.Signature>`_ ,
along with the arguments passed to ``dataprob.setup``, to determine how to treat
each parameter. 

Consider wrapping a function ``my_func``:

.. code-block:: python

    def my_func(a=7,b=1,c="test",d=1): 
        # do stuff here
        return some_1d_numpy_array

    f = dataprob.setup(my_func)

The software will assign parameters ``a`` and ``b`` as fittable, setting the
guesses to their default arguments (``a = 7`` and ``b = 1``). The ``c`` and
``d`` arguments will be set as non-fittable. This is because the default 
argument to ``c`` is not a float (``"test"``), and ``d`` occurs after a non-float
argument. In general, ``dataprob.setup`` grabs the first ``N`` arguments whose
default is a ``float`` or ``None``. All remaining arguments are treated as
non-fittable parameters. Fittable parameters with no default are assigned 
initial guesses of 0. 

Users can modify the default behavior with other arguments to
``dataprob.setup``. The ``fit_parameters`` argument can be used to directly
declare the fittable parameters. For example:

.. code-block:: python

    def my_func(a=7,b=1,c="test",d=1): 
        # do stuff here
        return some_1d_numpy_array

    f = dataprob.setup(my_func,
                       fit_parameters=['a','d'])
    
In this case, ``a`` and ``d`` will be fittable and ``b`` and ``c`` will be
non-fittable. Fit parameters can also be passed as a dictionary declaring 
guesses. For example ``fit_parameters={"a":8,"b":16,"d":-1}`` would set ``a``, 
``b`` and ``d`` to fittable, assigning their initial guesses as ``8``, ``16``,
and ``-1``. (Even more information can be passed in via ``fit_parameters``; 
see the :ref:`<fit_parameters> fit-param-ref` section below.)

.. note:: 
  
  If ``fit_parameters`` is specified, *only* the parameters listed in
  ``fit_parameters`` are fittable; all other parameters are non-fittable. 

The ``non_fit_kwargs`` argument plays the opposite role to ``fit_parameters``, 
allowing the user to declare non-fittable parameters and set their values.
For example: 

.. code-block:: python

    def my_func(a=7,b=1,c="test",d=1): 
        # do stuff here
        return some_1d_numpy_array

    f = dataprob.setup(my_func,
                       non_fit_kwargs={"a":5})
 
In this case, only ``b`` will be fittable, while ``a``, ``c``, and ``d`` will
be non-fittable, with values ``a = 5``, ``c = "test"``, and ``d = 1``. 



`fit_parameters` can differ from the parameters in the signature of `my_func` in
two cases: 

1.  If the signature of `my_func` contains `**kwargs`, `fit_parameters`
    can be used to specify parameters to pass into `my_func` that are
    not explicitly defined in the function signature. For example:

    .. code-block:: python

        def my_func(a,**kwargs): 
          # do stuff here
          return some_1d_numpy_array

        mw = dataprob.wrap_function(my_func,fit_parameters=['a','b','c'])
        
        # under the hood, dataprob will makes calls like:
        mw.model(a=a_value,b=b_value,c=c_value)

    In this case, the `a`, `b` and `c` parameters would be passed in as
    keyword arguments when the model is called. (The code does not check whether
    `my_func` can take those keyword arguments; that is the user's
    responsibility). 

2.  If `vector_first_arg` is `True`, `fit_parameters` defines the parameters
    to pass in as a numpy.ndarray as the first function argument. This works
    for functions with the following form: `my_func(some_array_arg,a,b)`, 
    where `some_array_arg` is numpy array argument that `some_func` knows what
    to do with. 

    .. code-block:: python

        def my_func(some_array_arg,a,b=1):
          # do stuff here
          return some_1d_numpy_array
        
        mw = dataprob.wrap_function(my_func,fit_parameters=['x','y','z'])
        
        # under the hood, dataprob will make calls like:
        mw.model(np.array([x_value,y_value,z_value]),a_value,b_value)
    
    If `vector_first_arg` is `True`, `fit_parameters` is required. All 
    function arguments besides this vector (`a` and `b` in this example) are
    treated as non-fittable parameters. 

.. _fit-param-ref:

The ``fit_parameters`` argument
-------------------------------

``fit_parameters`` can be used to declare more than just parameter names. It
can be used to set parameter guesses, whether or not they are fixed during the
regression, bounds, and gaussian priors. ``fit_parameters`` can be one of five
different types:

+ ``list``. Each entry is the name of the parameter as a string (e.g. ``['a','b']``).

+ ``dict`` with ``float`` values. The keys are the parameter names; the values
  are the parameter guesses (e.g. ``{'a':5,'b':11}``). 
  
+ ``dict`` with ``dict`` values. The keys are the parameter names; the values 
  are dictionaries keying parameter attributes to their values. For example:

  .. code-block:: python

      fit_parameters = {"a":{"guess":1,"lower_bound":0},
                        "b":{"upper_bound":20}`

  This indicates that parameter ``a`` should have a guess of ``1`` and a
  lower bound of zero. Parameter ``b`` should have an upper bound of ``20``.
  Note that the  dictionary does not need to exhaustively define all parameter
  features. Any parameter features that not specified are assigned defaults. 

+ ``dataframe``. The dataframe must have a ``name`` column with parameter names 
  (this corresponds directly to the parameter names in a ``fit_parameters``
  list). Other allowed columns are ``guess``, ``lower_bound``, ``upper_bound``,
  ``fixed``, ``prior_mean``, and ``prior_std``. These are described fully in the
  :ref:`<Parameters> full-parameters-ref` section above. 
    
+ ``string``: The software will treat this as a filename and will attempt to load
  it in as a dataframe (``xlsx``, ``csv``, and ``tsv`` are recognized.)
    
Samples
=======

Sample description here. 

