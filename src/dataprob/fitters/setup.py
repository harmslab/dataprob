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
    
    Basic pattern: 

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