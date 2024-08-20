"""
Public constructor used to set up analyses.
"""

from dataprob.fitters.ml import MLFitter
from dataprob.fitters.bootstrap import BootstrapFitter
from dataprob.fitters.bayesian.bayesian_sampler import BayesianSampler

def setup(some_function,
          method="ml",
          fit_parameters=None,
          non_fit_kwargs=None,
          vector_first_arg=False):
    """
    Set up a dataprob analysis. 

    Parameters
    ----------
    some_function : callable
        A function that takes at least one argument and returns a float numpy
        array. Fitter objects will compare the outputs of this function against
        y_obs. 
    method : str, default="ml"
        analysis method to use. should be "ml" (maximum likelihood), "bootstrap"
        (ml with bootstrap resampling of observation uncertainty), or "mcmc" 
        (Bayesian Markov Chain Monte Carlo sampling). 
    fit_parameters : list, dict, str, pandas.DataFrame; optional
        fit_parameters lets the user specify information about the parameters 
        in the fit. See Note below for details.
    non_fit_kwargs : dict
        non_fit_kwargs are keyword arguments for some_function that should not
        be fit but need to be specified to non-default values. 
    vector_first_arg : bool, default=False
        If True, the first argument of the function is taken as a vector of 
        parameters to fit. All other arguments to some_function are treated as 
        non-fittable parameters. Fit_parameters must then specify the names of
        each vector element. 

    Returns
    -------
    f : Fitter
        fitter instance to use for the analysis

    Notes
    -----
    
    **Basic pattern**

    ..code-block :: python

        # assume we have a dataframe named 'df' with the columns 'x', 'y_obs'
        # and 'y_std'. x is our independent variable, y_obs is what we observed
        # and y_std is the uncertainty on each observation.

        import dataprob

        # Define model
        def linear_model(m,b,x): return m*x + b

        # Set up the fit, passing in "x", which we need to run the model but is not
        # a fittable parameter. 
        f = dataprob.setup(linear_model,
                           non_fit_kwargs={"x":df["x"]})

        # do fit
        f.fit(y_obs=df["y_obs"],
              y_std=df["y_std"])

        # Plot and print fit result dataframe
        fig = dataprob.plot.plot_summary(f)
        print(f.fit_df)

    **Parameter setup**

    There are two classes of parameters to each model. Fittable parameters are
    visible to Fitter instances (such as the ML fitter or Bayesian sampler) and
    are thus regressed/sampled. Non-fittable parameters are fixed and passed
    into ``some_function`` whenever it is called, but are invisible to the Fitter. 

    The software uses the signature of ``some_function`, ``fit_parameters`, and
    ``vector_first_arg`` to figure out what fit parameters to use. 
    
    In the simplest case (`fit_parameters is None`, ``vector_first_arg is False`),
    the software will infer the fittable and non-fittable parameters from the
    ``some_function`` signature. It will grab the first N arguments with no
    default or whose default can be coerced to a float. The remaining arguments
    are treated as non-fittable parameters. Consider the example:

        ``some_function == my_func(a,b=1,c="test",d=1)`

    The software will find the fittable parameters ``a`` and ``b`, setting the
    guesses to ``a = 0`` and ``b = 1`. The ``c`` and ``d`` parameters will be set as
    non-fittable.  
    
    If fittable_parameters is defined, it can override this default. For 
    example, if ``fit_parameters = ['a','d']`, ``a`` and ``d`` will be fittable
    parameters and ``b`` and ``c`` will be non-fittable parameters. Except for two
    special cases described below, the parameters in ``fit_parameters`` must match
    the parameters in the function signature. The parameters ``a`, ``b`, and ``d`` 
    can be specified as fittable; the parameter ``c`` cannot because its default
    argument is a string. 

    NOTE: ``fit_parameters`` is treated as an exhaustive list of fittable 
    parameters. If specified, *only* the parameters in the list will be
    fittable.

    ``fit_parameters`` can differ from the parameters in the signature of 
    ``some_function`` in two cases: 
    
    1)  If the signature of ``some_function`` contains ``**kwargs`, ``fit_parameters`
        can be used to specify parameters to pass into some_function that are
        not explicitly delineated in the function signature. For example:

            ``some_function == my_func(a,**kwargs)`
    
        would allow ``fit_parameters = ['a','b','c']`. The ``b`` and ``c`` parameters 
        would be passed in as keyword arguments. (The code does not check 
        whether ``my_func`` can take those keyword arguments; that is the user's
        responsibility) 

    2)  If ``vector_first_arg`` is ``True`, ``fit_parameters`` defines the parameters
        to pass in as a numpy.ndarray as the first function argument. If
        ``vector_first_arg`` is ``True`, ``fit_parameters`` is required. All 
        function arguments besides this vector are treated as non-fittable 
        parameters. 
    
    Finally, ``fit_parameters`` can be used to pass in other information about 
    the fit parameters. This includes the parameter guess, whether or not it is
    fixed during the regression, its bounds, and the mean and standard deviation
    of a gaussian prior to apply to that fit parameter (Bayesian sampling only).
    This information can either be passed in via a dictionary or dataframe. 

    If ``fit_parameters`` comes in as a dataframe, the dataframe must have a 
    ``name`` column with parameter names (just like the entries to a
    ``fit_parameters`` list). It may have entries as described in the table below.
        
    If ``fit_parameters`` comes in as a dictionary, the keys should be the
    parameter names (just like the entries to a ``fit_parameters`` list). The
    values should be dictionaries keying parameter attributes to their values.
    For example:

        ``fit_parameters = {"K":{"guess":1,"lower_bound":0}}`
    
    would indicate that parameter "K" should have a guess of 1 and a lower bound
    of zero. 

    If ``fit_parameters`` comes in as a string, the software will treat this as 
    a filename and will attempt to load it in as a dataframe.    
    
    The allowed columns (for the dataframe) or keys (for the dictionary) are: 

        +---------------+-----------------------------------------------------+
        | key           | value                                               |
        +===============+=====================================================+
        | 'guess'       | guess as single float value (must be non-nan and    |
        |               | within bounds if specified)                         |
        +---------------+-----------------------------------------------------+
        | 'fixed'       | whether or not parameter can vary. True of False    |
        +---------------+-----------------------------------------------------+
        | 'lower_bound' | single float value; -np.inf allowed; None, nan or   |
        |               | pd.NA interpreted as -np.inf.                       |
        +---------------+-----------------------------------------------------+
        | 'upper_bound' | single float value; -np.inf allowed; None, nan or   |
        |               | pd.NA interpreted as np.inf.                        |
        +---------------+-----------------------------------------------------+
        | 'prior_mean'  | single float value; np.nan allowed (see below)      |
        +---------------+-----------------------------------------------------+
        | 'prior_std'   | single float value; np.nan allowed (see below)      |
        +---------------+-----------------------------------------------------+

        Gaussian priors are specified using the 'prior_mean' and 'prior_std' 
        fields, declaring the prior mean and standard deviation. If both are
        set to nan for a parameter, the prior is set to uniform between the
        parameter bounds. If either 'prior_mean' or 'prior_std' is set to a
        non-nan value, both must be non-nan to define the prior. When set, 
        'prior_std' must be greater than zero. Neither can be np.inf.         
    """

    
    method_map = {"ml":MLFitter,
                  "bootstrap":BootstrapFitter,
                  "mcmc":BayesianSampler}
    
    if method not in method_map:
        err = "method should be one of:\n"
        for k in method_map:
            err += f"    {k}\n"
        raise ValueError(err)
    
    return method_map[method](some_function=some_function,
                              fit_parameters=fit_parameters,
                              non_fit_kwargs=non_fit_kwargs,
                              vector_first_arg=vector_first_arg)