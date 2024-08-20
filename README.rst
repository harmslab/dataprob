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
likelihood fit to estimate the slope and intercept. You can change the type of
analysis by changing the definition of `f`.

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

    # Create an MLFitter (or Bayesian or Bootstrap...)
    f = dataprob.MLFitter(some_function=linear_model,
                          non_fit_kwargs={"x":x_array})
    
    #f = dataprob.BayesianSampler(some_function=linear_model,
    #                             non_fit_kwargs={"x":x_array})
    
    #f = dataprob.BootstrapFitter(some_function=linear_model,
    #                             non_fit_kwargs={"x":x_array})

    # Fit the wrapped model to y_obs, setting our estimated uncertainty
    # on each observed point to 0.5
    f.fit(y_obs=y_obs,
          y_std=0.5)

    f.fit_df

The `f.fit_df` dictionary will look something like:

+-------+------+----------+-------+--------+---------+-------+---------+-------------+-------------+------------+-----------+
| index | name | estimate | std   | low_95 | high_95 | guess | fixed   | lower_bound | upper_bound | prior_mean | prior_std |
+=======+======+==========+=======+========+=========+=======+=========+=============+=============+============+===========+
| `m`   | `m`  | 5.009    |	0.045 | 4.817  | 5.202   | 1.0   | `False` | `-inf`      | `inf`       | `NaN`      | `NaN`     |  
+-------+------+----------+-------+--------+---------+-------+---------+-------------+-------------+------------+-----------+
| `b`   | `b`  | 5.644    |	0.274 |	4.465  | 6.822   | 1.0   | `False` | `-inf`      | `inf`       | `NaN`      | `NaN`     |
+-------+------+----------+-------+--------+---------+-------+---------+-------------+-------------+------------+-----------+


Fitters
=======

There are three different analyses possible:

+ *MLFitter*: Does a maximum likelihood fit, regressing model parameters against
  observed data. 
+ *BayesianSampler*: Uses Markov-Chain Monte Carlo to generate the posterior
  distributions for model parameters. 
+ *BootstrapFitter*: Estimates parameter distributions consistent with 
  observed data by sampling observation uncertainty and using maximum likelihood
  to fit the model to each pseudoreplicate dataset. 

Parameters
==========

Parameters are defined in the `f.param_df` dataframe. 

The 'name' column is set when the dataframe is initialized. This defines
the names of the parameters, which cannot be changed later. The 'name'
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

The param_df will have the following columns. Other columns may be present if
set by the user, but will be ignored. 

+---------------+-----------------------------------------------------+
| key           | value                                               |
+===============+=====================================================+
| `name`        | string name of the parameter. should not be changed |
|               | by the user once fitter is initialized.             |
+---------------+-----------------------------------------------------+
| `guess`       | guess as single float value (must be non-nan and    |
|               | within bounds if specified)                         |
+---------------+-----------------------------------------------------+
| `fixed`       | whether or not parameter can vary. `True` of `False`|
+---------------+-----------------------------------------------------+
| `lower_bound` | single float value; `-np.inf` allowed; `None`, `nan`|
|               | or `pd.NA` interpreted as `np.inf`.                 |
+---------------+-----------------------------------------------------+
| `upper_bound` | single float value; `-np.inf` allowed; `None`, `nan`|
|               | or `pd.NA` interpreted as `np.inf`.                 |
+---------------+-----------------------------------------------------+
| `prior_mean`  | single float value; `np.nan` allowed (see below)    |
+---------------+-----------------------------------------------------+
| `prior_std`   | single float value; `np.nan` allowed (see below)    |
+---------------+-----------------------------------------------------+

Gaussian priors are specified using the `prior_mean` and `prior_std` fields, 
declaring the prior mean and standard deviation. If both are set to `nan` for a
parameter, the prior for that parameter is set to uniform between the parameter
bounds. If either `prior_mean` or `prior_std` is set to a non-nan value, both
must be non-nan to define the prior. When set, `prior_std` must be greater than
zero. Neither can be `np.inf`. Both a gaussian prior and bounds may be
specified. 

Model definition
================

The software can wrap and regress the parameters to any function that: 

1. Has at least one numerical argument

2. Returns a numpy array the same length as `y_obs`. 

The function can be a simple function, method of a complicated class, or any
other object with a `__call__` attribute.

There are two types of parameters for each model. Fittable parameters are
visible to Fitter instances (such as the ML fitter or Bayesian sampler) and
are thus regressed/sampled. Non-fittable parameters are fixed and passed
into the wrapped function whenever it is called, but are invisible to the
Fitters. 

Consider wrapping a function `my_func`. The software uses the 
`signature <https://docs.python.org/3/library/inspect.html#inspect.Signature>`_ 
of the function, as well as two other arguments, `fit_parameters` and
`vector_first_arg`, to figure out what fit parameters to use. 

In the simplest case (`fit_parameters is None`, `vector_first_arg is False`),
the software infers the fittable and non-fittable parameters from the
signature of `my_func`. It grabs the first N arguments with no
default or whose default can be coerced to a float. The remaining arguments
are treated as non-fittable parameters. Consider the example:

.. code-block:: python

    def my_func(a,b=1,c="test",d=1): 
        # do stuff here
        return some_1d_numpy_array

    mw = dataprob.wrap_function(my_func)

The software will find the fittable parameters `a` and `b`, setting the
guesses to `a = 0` and `b = 1`. The `c` and `d` parameters will be set as
non-fittable.  

If `fittable_parameters`` is defined, it can override this default. For 
example:

.. code-block:: python

    def my_func(a,b=1,c="test",d=1): 
        # do stuff here
        return some_1d_numpy_array

    mw = dataprob.wrap_function(my_func,fit_parameters=['a','d'])
    
In this case, `a` and `d` will be fittable parameters and `b` and `c` will
be non-fittable parameters. Except for two special cases described below, the
parameters in `fit_parameters` must match the parameters in the function
signature. The parameters `a`, `b`, and `d` can be specified as fittable 
because they either have no default (`a`) or numeric defaults (`b` and `d`). 
The parameter `c` cannot be fittable because its default argument is a string. 

.. note::

  `fit_parameters` is used as an exhaustive list of fittable parameters. If
  specified, *only* the parameters in the list will be fittable.

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

fit_parameters argument
-----------------------

In addition to specifying the names of the fittable parameters, `fit_parameters`
can be used to pass in other information about the parameters. This includes the
parameter guess, whether or not it is fixed during the regression, its bounds,
and the mean and standard deviation of a gaussian prior to use for that 
parameter in a Bayesian MCMC analysis. `fit_paramters` can be four different
types:

+ `list`: each entry is the name of the parameter as a string (e.g. `['a','b']`).

+ `dict`: the keys should be the parameter names (just like the entries in a
  `fit_parameters` list). The values should be dictionaries keying parameter
  attributes to their values. For example:

  .. code-block:: python

      fit_parameters = {"a":{"guess":1,"lower_bound":0},
                        "b":{"upper_bound":20}`

  This indicates that parameter `a` should have a guess of 1 and a lower bound
  of zero. Parameter `b` should have an upper bound of 20. Note that the 
  dictionary does not need to exhaustively define all parameter features. Any
  values that not specified are assigned defaults. 

+ `dataframe`: the dataframe must have a `name` column with parameter names 
  (just like the entries in a `fit_parameters` list). Other allowed columns are
  `guess`, `lower_bound`, `upper_bound`, `fixed`, `prior_mean`, and `prior_std`. 
  These are described fully in the *Parameters* section above. 
    
+ `string`: the software will treat this as a filename and will attempt to load
  it in as a dataframe (`xlsx`, `csv`, and `tsv` are recognized.)
    
Samples
=======

