========
dataprob
========

Library for calculating the probability of data (the likelihood) to extract
parameter estimates for models describing experimental data. Can do maximum
likelihood, bootstrap, and Bayesian sampling using a consistent API.  

Docs in progress.  See `examples/example-fit.ipynb` for basic demonstration
of the API.

Fitters
=======

+ :emph:`MLFitter`: Does a maximum likelihood fit, regressing model
  parameters against observed data. 
+ :emph:`BayesianSampler`: Performs Markov-Chain Monte Carlo to generate 
  posterior distributions for model parameters. 


Parameters
==========

Dataframe holding the fittable parameters in the model. This can be set 
by ``mw.param_df = some_new_df``. It can also be edited in place 
(e.g. ``mw.param_df.loc["K1","guess"] = 5``).



The 'name' column is set when the dataframe is initialized. This defines
the names of the parameters, which cannot be changed later. The 'name'
column is used as the index for the dataframe. 

This dataframe will minimally have the following columns. Other
columns may be present if set by the user, but will be ignored. 

+---------------+-----------------------------------------------------+
| key           | value                                               |
+===============+=====================================================+
| 'name'        | string name of the parameter. should not be changed |
|               | by the user.                                        |
+---------------+-----------------------------------------------------+
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

