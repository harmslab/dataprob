
import pytest

import numpy as np
import pandas as pd
from scipy import stats
import emcee

from dataprob.fitters.bayesian.bayesian_sampler import BayesianSampler
from dataprob.fitters.bayesian._prior_processing import find_normalization
from dataprob.fitters.bayesian._prior_processing import reconcile_bounds_and_priors
from dataprob.fitters.bayesian._prior_processing import find_uniform_value

import warnings

def test_BayesianSampler__init__():

    def test_fcn(a,b): return a*b

    # default args work. check to make sure super().__init__ actually ran.
    f = BayesianSampler(some_function=test_fcn)
    assert f.num_obs is None

def test__setup_priors():

    # Two parameter test function to wrap
    def test_fcn(a=1,b=2): return a*b

    # ----------------------------------------------------------------------
    # basic functionality with a uniform and gaussian prior

    f = BayesianSampler(some_function=test_fcn)
    assert not hasattr(f,"_prior_frozen_rv")
    assert not hasattr(f,"_uniform_priors")
    assert not hasattr(f,"_gauss_prior_means")
    assert not hasattr(f,"_gauss_prior_stds")
    assert not hasattr(f,"_gauss_prior_offsets")
    assert not hasattr(f,"_gauss_prior_mask")
    assert not hasattr(f,"_lower_bounds")
    assert not hasattr(f,"_upper_bounds")

    # Load model and set priors & bounds
    f.param_df["prior_mean"] = [0,np.nan]
    f.param_df["prior_std"] = [1,np.nan]
    f.param_df["lower_bound"] = [-np.inf,-np.inf]
    f.param_df["upper_bound"] = [np.inf,np.inf]
    f._model.finalize_params()

    f._setup_priors()

    assert f._prior_frozen_rv is not None
    assert np.isclose(stats.norm(loc=0,scale=1).logpdf(0),
                      f._prior_frozen_rv.logpdf(0))
    assert issubclass(type(f._uniform_priors),float)
    assert f._uniform_priors < 0

    assert len(f._gauss_prior_means) == 1
    assert f._gauss_prior_means[0] == 0
    assert len(f._gauss_prior_stds) == 1
    assert f._gauss_prior_stds[0] == 1
    assert len(f._gauss_prior_offsets) == 1
    assert f._gauss_prior_offsets[0] < 0
    assert np.array_equal(f._gauss_prior_mask,[True,False])

    # ----------------------------------------------------------------------
    # No gaussian priors

    # Load model and set priors & bounds
    f = BayesianSampler(some_function=test_fcn)
    f.param_df["prior_mean"] = [np.nan,np.nan]
    f.param_df["prior_std"] = [np.nan,np.nan]
    f.param_df["lower_bound"] = [-np.inf,-np.inf]
    f.param_df["upper_bound"] = [np.inf,np.inf]
    f._model.finalize_params()

    f._setup_priors()

    assert len(f._gauss_prior_means) == 0
    assert len(f._gauss_prior_stds) == 0
    assert len(f._gauss_prior_offsets) == 0
    assert np.array_equal(f._gauss_prior_mask,[False,False])

    # ----------------------------------------------------------------------
    # No uniform priors

    # Load model and set priors & bounds
    f = BayesianSampler(some_function=test_fcn)
    f.param_df["prior_mean"] = [1.0,2.0]
    f.param_df["prior_std"] = [3.0,4.0]
    f.param_df["lower_bound"] = [-np.inf,-np.inf]
    f.param_df["upper_bound"] = [np.inf,np.inf]
    f._model.finalize_params()

    f._setup_priors()

    assert np.isclose(f._uniform_priors,0)
    assert np.allclose(f._gauss_prior_means,[1.0,2.0])
    assert np.allclose(f._gauss_prior_stds,[3.0,4.0])
    assert len(f._gauss_prior_offsets) == 2
    assert np.sum(f._gauss_prior_offsets < 0) == 2
    assert np.array_equal(f._gauss_prior_mask,[True,True])

    # ----------------------------------------------------------------------
    # Two gauss, two uniform, one of each fixed

    def four_param(a,b,c,d): return a*b*c*d

    # Load model and set priors & bounds
    f = BayesianSampler(some_function=four_param)
    f.param_df["prior_mean"] = [1.0,2.0,np.nan,np.nan]
    f.param_df["prior_std"] = [3.0,4.0,np.nan,np.nan]
    f.param_df["lower_bound"] = [-np.inf,-np.inf,-np.inf,-np.inf]
    f.param_df["upper_bound"] = [np.inf,np.inf,np.inf,np.inf]
    f.param_df["fixed"] = [True,False,True,False]
    f._model.finalize_params()

    f._setup_priors()

    assert np.allclose(f._gauss_prior_means,[2.0])
    assert np.allclose(f._gauss_prior_stds,[4.0])
    assert len(f._gauss_prior_offsets) == 1
    assert np.sum(f._gauss_prior_offsets < 0) == 1
    assert np.array_equal(f._gauss_prior_mask,[True,False])


    # ----------------------------------------------------------------------
    # check internal bounds calculation adjustment calculation

    # Load model and set priors & bounds
    def single_param(a): return a
    f = BayesianSampler(some_function=single_param)
    f.param_df["prior_mean"] = [10.0]
    f.param_df["prior_std"] = [5.0]
    f.param_df["lower_bound"] = [0.0]
    f.param_df["upper_bound"] = [20.0]
    f._model.finalize_params()

    f._setup_priors()

    # Check parsing/basic run
    assert np.isclose(f._uniform_priors,0)
    assert np.array_equal(f._gauss_prior_means,[10])
    assert np.array_equal(f._gauss_prior_stds,[5])
    assert np.array_equal(f._gauss_prior_mask,[True])
    
    # Make sure final offset from the code matches what we calculate here. (Not
    # really testing math bit -- that's in the _find_normalization and 
    # _reconcile_bounds_and_priors tests).
    base_offset = find_normalization(scale=1,rv=stats.norm)
    bounds = np.array([0,20])
    bounds_offset = reconcile_bounds_and_priors(bounds=(bounds - 10)/5,
                                                frozen_rv=stats.norm(loc=0,scale=1))
    assert np.isclose(base_offset + bounds_offset,f._gauss_prior_offsets[0])

    # Make sure the code is really pulling the bounds from the param_df (needed
    # for fast prior calcs).
    assert np.array_equal(f._lower_bounds,f.param_df["lower_bound"])
    assert np.array_equal(f._upper_bounds,f.param_df["upper_bound"])

def test_BayesianSampler__ln_prior():

    # ----------------------------------------------------------------------
    # single parameter priors, bounded, numerical test

    # Load model and set priors & bounds
    def single_param(a): return a
    f = BayesianSampler(some_function=single_param)
    f.param_df["prior_mean"] = [0]
    f.param_df["prior_std"] = [1]
    f.param_df["lower_bound"] = [-1]
    f.param_df["upper_bound"] = [1]
    f._model.finalize_params()
    f._setup_priors()

    # Set up local calculation
    frozen_rv = stats.norm(loc=0,scale=1)
    base_offset = find_normalization(scale=1,rv=stats.norm)
    bounds = np.array([-1,1])
    bounds_offset = reconcile_bounds_and_priors(bounds=(bounds-0)/1,
                                                frozen_rv=frozen_rv)
    
    # Try a set of values
    for v in [-0.9,-0.1,0.0,0.1,0.9]:

        print("testing",v)
        expected = frozen_rv.logpdf(v) + base_offset + bounds_offset
        value = f._ln_prior(param=np.array([v]))
        assert np.isclose(expected,value)

    # Go outside of bounds
    expected = -np.inf
    value = f._ln_prior(param=np.array([2]))
    assert np.isclose(expected,value)

    expected = -np.inf
    value = f._ln_prior(param=np.array([-2]))
    assert np.isclose(expected,value)

    # ----------------------------------------------------------------------
    # Now set up two priors, one gauss with one infinite bound, one uniform with
    # infinte bound

    def two_parameter(a=1,b=2): return a*b
    f = BayesianSampler(some_function=two_parameter)
    f.param_df["guess"] = [-5,-5]
    f.param_df["prior_mean"] = [2,np.nan]
    f.param_df["prior_std"] = [10,np.nan]
    f.param_df["lower_bound"] = [-np.inf,-np.inf]
    f.param_df["upper_bound"] = [10,0]
    f._model.finalize_params()
    f._setup_priors()

    # Set up local calculation
    frozen_rv = stats.norm(loc=0,scale=1)
    base_offset = find_normalization(scale=1,rv=stats.norm)
    bounds = np.array([-np.inf,10])
    bounds_offset = reconcile_bounds_and_priors(bounds=(bounds-2)/10,
                                                frozen_rv=frozen_rv)
    
    total_gauss_offset = base_offset + bounds_offset
    uniform_prior = find_uniform_value(bounds=np.array([-np.inf,0]))

    for v in [-1e6,-1,0,8]:
        print("testing",v)
        z = (v - 2)/10
        expected = frozen_rv.logpdf(z) + total_gauss_offset + uniform_prior
        value = f._ln_prior(np.array([v,-5]))
        assert np.isclose(expected,value)

    # Outside of bounds
    value = f._ln_prior(np.array([-1,2]))
    assert np.isclose(-np.inf,value)

def test_BayesianSampler_ln_prior():
    
    # test error checking. __ln_prior test checks numerical results 
    
    def linear_fcn(m,b,x): return m*x + b
    x = np.linspace(-5,5,10)
    expected_result = np.array([2,-1])

    f = BayesianSampler(some_function=linear_fcn,
                        non_fit_kwargs={"x":x})
    
    v = f.ln_prior(expected_result)
    assert v < 0

    with pytest.raises(ValueError):
        f.ln_prior(["a","b"])

    with pytest.raises(ValueError):
        f.ln_prior([1])

    with pytest.raises(ValueError):
        f.ln_prior([1,2,3])


def test_BayesianSampler__ln_prob():

    # Not really a numeric test, but makes sure the code is in fact summing 
    # ln_prior and ln_like and recognizing nan/inf correctly

    def linear_fcn(m,b,x): return m*x + b
    x = np.linspace(-5,5,10)
    data_df = pd.DataFrame({"y_obs":linear_fcn(m=2,b=-1,x=x),
                            "y_std":0.1*np.ones(10)})
    expected_result = np.array([2,-1])

    f = BayesianSampler(some_function=linear_fcn,
                        non_fit_kwargs={"x":x})
    f.data_df = data_df

    f.param_df["guess"] = expected_result
    f.param_df["prior_mean"] = [2,1]
    f.param_df["prior_std"] = [2,2]
    f.param_df["lower_bound"] = [-np.inf,-np.inf]
    f.param_df["upper_bound"] = [np.inf,np.inf]
    f._model.finalize_params()
    f._setup_priors()

    ln_like = f.ln_like(expected_result)
    ln_prob = f.ln_prior(expected_result)

    assert f._ln_prob(expected_result) == ln_like + ln_prob
    assert np.isinf(f._ln_prob(np.array([np.nan,1])))
    
def test_BayesianSampler_ln_prob():
    
    # test error checking. __ln_prob test checks numerical results 
    
    def linear_fcn(m,b,x): return m*x + b
    x = np.linspace(-5,5,10)
    data_df = pd.DataFrame({"y_obs":linear_fcn(m=2,b=-1,x=x),
                            "y_std":0.1*np.ones(10)})
    expected_result = np.array([2,-1])

    f = BayesianSampler(some_function=linear_fcn,
                        non_fit_kwargs={"x":x})
    
    # should not work -- no y_obs, y_std loaded
    with pytest.raises(RuntimeError):
        f.ln_prob(expected_result)
    
    f.data_df = data_df

    # should work -- all needed features defined
    v = f.ln_prob(expected_result)
    assert v < 0
    
    with pytest.raises(ValueError):
        f.ln_prob(["a","b"])

    with pytest.raises(ValueError):
        f.ln_prob([1])

    with pytest.raises(ValueError):
        f.ln_prob([1,2,3])

def test_BayesianSampler__sample_to_convergence():

    def test_fcn(m,b,x): return m*x + b

    # ------
    # test no convergence 

    f = BayesianSampler(some_function=test_fcn,
                        non_fit_kwargs={"x":np.arange(10)})
    f._y_obs = np.arange(10)*5 + 2 + np.random.normal(0,1,10)
    f._y_std = 1.0

    num_walkers = 100
    num_steps = 100

    f._setup_priors()
    f._initial_state = np.ones((100,2))
    f._initial_state[:,0] = np.random.normal(5,0.1,100)
    f._initial_state[:,1] = np.random.normal(2,0.1,100)
        
    # Build sampler object
    f._fit_result = emcee.EnsembleSampler(nwalkers=num_walkers,
                                          ndim=f._initial_state.shape[1],
                                          log_prob_fn=f._ln_prob)
    f._num_steps = num_steps

    # Cannot converge -- check for warning and setting success to False
    f._max_convergence_cycles = 1
    with pytest.warns():
        f._sample_to_convergence()
    assert f._success is False

    # ------
    # test convergence
    f = BayesianSampler(some_function=test_fcn,
                        non_fit_kwargs={"x":np.arange(10)})
    f._y_obs = np.arange(10)*5 + 2 + np.random.normal(0,1,10)
    f._y_std = 1.0

    num_walkers = 100
    num_steps = 100

    f._setup_priors()
    f._initial_state = np.ones((100,2))
    f._initial_state[:,0] = np.random.normal(5,0.1,100)
    f._initial_state[:,1] = np.random.normal(2,0.1,100)
        
    # Build sampler object
    f._fit_result = emcee.EnsembleSampler(nwalkers=num_walkers,
                                          ndim=f._initial_state.shape[1],
                                          log_prob_fn=f._ln_prob)
    f._num_steps = num_steps

    # converges -- no warning, set to False
    f._max_convergence_cycles = 10
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        f._sample_to_convergence()
    assert f._success is True




def test_BayesianSampler_fit():

    def test_fcn(m,b,x): return m*x + b

    f = BayesianSampler(some_function=test_fcn,
                        non_fit_kwargs={"x":np.arange(10)})
    y_obs = np.arange(10)*1 + 2
    y_std = 1.0

    f.fit(y_obs=y_obs,
          y_std=y_std,
          num_walkers=10,
          use_ml_guess=True,
          num_steps=100,
          burn_in=0.1,
          max_convergence_cycles=10,
          num_threads=1)

    assert f._num_walkers == 10
    assert f._use_ml_guess is True
    assert f._num_steps == 100
    assert np.isclose(f._burn_in,0.1)
    assert f._num_threads == 1
    assert f._max_convergence_cycles == 10
    
    # check num threads passing
    with pytest.raises(NotImplementedError):
        f.fit(y_obs=y_obs,
              y_std=y_std,
              num_walkers=10,
              use_ml_guess=True,
              num_steps=10,
              burn_in=0.1,
              num_threads=0)
        
    with pytest.raises(NotImplementedError):
        f.fit(y_obs=y_obs,
              y_std=y_std,
              num_walkers=10,
              use_ml_guess=True,
              num_steps=10,
              burn_in=0.1,
              num_threads=10)
    
    # Pass bad value into each kwarg to make sure checker is running
    with pytest.raises(ValueError):
        f.fit(y_obs=y_obs,
              y_std=y_std,
              num_walkers=0)        
    with pytest.raises(ValueError):
        f.fit(y_obs=y_obs,
              y_std=y_std,
              use_ml_guess="not a bool")
    with pytest.raises(ValueError):
        f.fit(y_obs=y_obs,
              y_std=y_std,
              num_steps=1.2)
    with pytest.raises(ValueError):
        f.fit(y_obs=y_obs,
              y_std=y_std,
              burn_in=0.0)
    with pytest.raises(ValueError):
        f.fit(y_obs=y_obs,
              y_std=y_std,
              num_threads=-2)

    with pytest.raises(ValueError):
        f.fit(y_obs=y_obs,
              y_std=y_std,
              max_convergence_cycles=0)

    
    with pytest.raises(TypeError):
        f.fit(y_obs=y_obs,
              y_std=y_std,
              not_an_emcee_kwarg="five")


def test_BayesianSampler__fit():
    
    def linear_fcn(m,b,x): return m*x + b
    x = np.linspace(-5,5,10)
    data_df = pd.DataFrame({"y_obs":linear_fcn(m=2,b=-1,x=x),
                            "y_std":0.1*np.ones(10)})

    # -------------------------------------------------------------------------
    # basic run; checking use_ml_guess = True effect

    f = BayesianSampler(some_function=linear_fcn,
                        non_fit_kwargs={"x":x})
    f.data_df = data_df

    assert np.array_equal(f.param_df["guess"],[0,0])
    assert not hasattr(f,"_initial_state")
    assert np.sum(np.isnan(f._fit_df["estimate"])) == 2
    assert f.fit_result is None
    assert f.samples is None

    # run containing fit function from base class; that sets fit_has_been_run to
    # true. Make sure containing function ran completely. 
    f.fit(num_walkers=100,
          use_ml_guess=True,
          num_steps=100,
          burn_in=0.1,
          num_threads=1,
          max_convergence_cycles=10)
    assert f._fit_has_been_run is True

    # These outputs are determined within ._fit
    assert issubclass(type(f._fit_result),emcee.ensemble.EnsembleSampler)
    assert f._initial_state.shape == (100,2)
    assert f._success is True
    assert np.sum(np.isnan(f.fit_df["estimate"])) == 0

    # -------------------------------------------------------------------------
    # basic run; checking use_ml_guess = False effect

    # Very small analysis, starting from ML
    f = BayesianSampler(some_function=linear_fcn,
                        non_fit_kwargs={"x":x})
    f.data_df = data_df

    # set guess to 1, 1. works, but differs from ML guess of 2, 1 and can thus
    # be distinguished below
    f.param_df["guess"] = [1,1]
    assert not hasattr(f,"_initial_state")
    assert np.sum(np.isnan(f._fit_df["estimate"])) == 2
    assert f.fit_result is None
    assert f.samples is None

    # run containing fit function from base class; that sets fit_has_been_run to
    # true. Make sure containing function ran completely. 
    f.fit(num_walkers=100,
          use_ml_guess=False,
          num_steps=100,
          burn_in=0.1,
          num_threads=1,
          max_convergence_cycles=10)
    assert f._fit_has_been_run is True

    # look for non-ML guess
    assert issubclass(type(f._fit_result),emcee.ensemble.EnsembleSampler)
    assert f._initial_state.shape == (100,2)
    assert f._success is True
    assert np.sum(np.isnan(f.fit_df["estimate"])) == 0

    # -------------------------------------------------------------------------
    # basic run; checking effects of altered num_steps and num_walkers

    # Very small analysis, starting from ML
    f = BayesianSampler(some_function=linear_fcn,
                        non_fit_kwargs={"x":x})
    f.data_df = data_df

    assert np.array_equal(f.param_df["guess"],[0,0])
    assert not hasattr(f,"_initial_state")
    assert np.sum(np.isnan(f._fit_df["estimate"])) == 2
    assert f.fit_result is None
    assert f.samples is None

    # run containing fit function from base class; that sets fit_has_been_run to
    # true. Make sure containing function ran completely. Warning comes about
    # because it will not converge.
    with pytest.warns():
        f.fit(num_walkers=9,
              use_ml_guess=True,
              num_steps=20,
              burn_in=0.1,
              num_threads=1,
              max_convergence_cycles=1)
    assert f._fit_has_been_run is True

    # These outputs are determined within ._fit
    assert issubclass(type(f._fit_result),emcee.ensemble.EnsembleSampler)
    assert f._initial_state.shape == (9,2)
    assert np.array_equal(f.samples.shape,[162,2]) 
    assert f._lnprob.shape == (162,)
    assert f._success is False
    assert np.sum(np.isnan(f.fit_df["estimate"])) == 0

    # -------------------------------------------------------------------------
    # basic run; altered burn in

    # Very small analysis, starting from ML
    f = BayesianSampler(some_function=linear_fcn,
                        non_fit_kwargs={"x":x})
    f.data_df = data_df

    assert np.array_equal(f.param_df["guess"],[0,0])
    assert not hasattr(f,"_initial_state")
    assert np.sum(np.isnan(f._fit_df["estimate"])) == 2
    assert f.fit_result is None
    assert f.samples is None

    # run containing fit function from base class; that sets fit_has_been_run to
    # true. Make sure containing function ran completely. Warning comes about
    # because it will not converge. 
    with pytest.warns():
        f.fit(num_walkers=10,
            use_ml_guess=True,
            num_steps=10,
            burn_in=0.5,
            num_threads=1,
            max_convergence_cycles=1)
    assert f._fit_has_been_run is True

    # These outputs are determined within ._fit
    assert issubclass(type(f._fit_result),emcee.ensemble.EnsembleSampler)
    assert f._initial_state.shape == (10,2)
    assert np.array_equal(f.samples.shape,[50,2]) 
    assert f._lnprob.shape == (50,)
    assert f._success is False
    assert np.sum(np.isnan(f.fit_df["estimate"])) == 0

    # -------------------------------------------------------------------------
    # basic run; fixed parameter; no ml

    # Very small analysis, starting from no ML
    f = BayesianSampler(some_function=linear_fcn,
                        non_fit_kwargs={"x":x})
    f.data_df = data_df
    
    f.param_df.loc["b","fixed"] = True
    f.param_df.loc["m","prior_mean"] = 2
    f.param_df.loc["m","prior_std"] = 5
    
    assert np.array_equal(f.param_df["guess"],[0,0])
    assert np.array_equal(f.param_df["fixed"],[False,True]) # make sure fixed
    assert not hasattr(f,"_initial_state")
    assert np.sum(np.isnan(f._fit_df["estimate"])) == 2
    assert f.fit_result is None
    assert f.samples is None

    # run containing fit function from base class; that sets fit_has_been_run to
    # true. Make sure containing function ran completely. 
    f.fit(num_walkers=100,
          use_ml_guess=False,
          num_steps=100,
          burn_in=0.1,
          num_threads=1,
          max_convergence_cycles=10)
    assert f._fit_has_been_run is True

    # These outputs are determined within ._fit
    assert issubclass(type(f._fit_result),emcee.ensemble.EnsembleSampler)
    assert f._initial_state.shape == (100,1)
    assert f._success is True
    assert np.sum(np.isnan(f.fit_df["estimate"])) == 0
    assert f.fit_df.loc["b","estimate"] == f.fit_df.loc["b","guess"]

    # -------------------------------------------------------------------------
    # basic run; fixed parameter; ml

    # Very small analysis, starting from ML
    f = BayesianSampler(some_function=linear_fcn,
                        non_fit_kwargs={"x":x})
    f.data_df = data_df

    f.param_df.loc["b","fixed"] = True

    assert np.array_equal(f.param_df["guess"],[0,0])
    assert np.array_equal(f.param_df["fixed"],[False,True]) # make sure fixed
    assert not hasattr(f,"_initial_state")
    assert np.sum(np.isnan(f._fit_df["estimate"])) == 2
    assert f.fit_result is None
    assert f.samples is None

    # run containing fit function from base class; that sets fit_has_been_run to
    # true. Make sure containing function ran completely. 
    f.fit(num_walkers=100,
          use_ml_guess=True,
          num_steps=100,
          burn_in=0.1,
          num_threads=1,
          max_convergence_cycles=10)
    assert f._fit_has_been_run is True

    # These outputs are determined within ._fit
    assert issubclass(type(f._fit_result),emcee.ensemble.EnsembleSampler)
    assert f._initial_state.shape == (100,1)
    assert f._success is True
    assert np.sum(np.isnan(f.fit_df["estimate"])) == 0
    assert f.fit_df.loc["b","estimate"] == f.fit_df.loc["b","guess"]

    # -------------------------------------------------------------------------
    # run twice in a row to check for sample appending

    # Very small analysis, starting from ML
    f = BayesianSampler(some_function=linear_fcn,
                        non_fit_kwargs={"x":x})
    f.data_df = data_df

    assert np.array_equal(f.param_df["guess"],[0,0])
    assert not hasattr(f,"_initial_state")
    assert np.sum(np.isnan(f._fit_df["estimate"])) == 2
    assert f.fit_result is None
    assert f.samples is None

    # run containing fit function from base class; that sets fit_has_been_run to
    # true. Make sure containing function ran completely. Warning comes about
    # because it will not have converged. 
    with pytest.warns():
        f.fit(num_walkers=10,
            use_ml_guess=True,
            num_steps=10,
            burn_in=0.1,
            num_threads=1,
            max_convergence_cycles=1)
    assert f._fit_has_been_run is True

    # These outputs are determined within ._fit
    assert issubclass(type(f._fit_result),emcee.ensemble.EnsembleSampler)
    assert f._initial_state.shape == (10,2)
    assert np.array_equal(f.samples.shape,[90,2]) 
    assert f._lnprob.shape == (90,)
    assert f._success is False
    assert np.sum(np.isnan(f.fit_df["estimate"])) == 0

    # now run again. Warning because not converged. 
    with pytest.warns():
        f.fit(num_walkers=10,
            use_ml_guess=True,
            num_steps=10,
            burn_in=0.1,
            num_threads=1,
            max_convergence_cycles=1)

    assert issubclass(type(f._fit_result),emcee.ensemble.EnsembleSampler)
    assert f._initial_state.shape == (10,2)
    assert np.array_equal(f.samples.shape,[180,2]) 
    assert f._lnprob.shape == (180,)
    assert f._success is False
    assert np.sum(np.isnan(f.fit_df["estimate"])) == 0

    # --------------------------------------------------------------------------
    # Run test where the ml fit will throw an error. Should be caught. 
    def bad_fcn(m,b,x): return np.ones(10)*np.nan
    x = np.linspace(-5,5,10)
    data_df = pd.DataFrame({"y_obs":linear_fcn(m=2,b=-1,x=x),
                            "y_std":0.1*np.ones(10)})

    f = BayesianSampler(some_function=bad_fcn,
                        non_fit_kwargs={"x":x})
    f.data_df = data_df
    with pytest.raises(RuntimeError):
        f.fit(use_ml_guess=True)

    # --------------------------------------------------------------------------
    # Run test where the ml fit will work, but the Jacobian will be singular
    # so it cannot generate samples 
    
    def bad_fcn(m,b,x): return np.ones(10)
    x = np.linspace(-5,5,10)
    data_df = pd.DataFrame({"y_obs":linear_fcn(m=2,b=-1,x=x),
                            "y_std":0.1*np.ones(10)})

    f = BayesianSampler(some_function=bad_fcn,
                        non_fit_kwargs={"x":x})
    f.data_df = data_df
    with pytest.raises(RuntimeError):
        with pytest.warns():
            f.fit(use_ml_guess=True)


def test_BayesianSampler__update_fit_df():
    
    # Create a BayesianSampler with a model loaded (and _fit_df implicitly 
    # created)
    
    def test_fcn(a=1,b=2): return a*b
    f = BayesianSampler(some_function=test_fcn)

    # add some fake samples
    f._samples = np.random.normal(loc=0,scale=1,size=(10000,2))

    assert np.sum(np.isnan(f._fit_df["estimate"])) == 2
    assert np.sum(np.isnan(f._fit_df["std"])) == 2
    assert np.sum(np.isnan(f._fit_df["low_95"])) == 2
    assert np.sum(np.isnan(f._fit_df["high_95"])) == 2

    f._update_fit_df()

    # Make sure mean/std/95 calc is write given fake samples we stuffed in
    assert np.allclose(np.round(f._fit_df["estimate"],0),[0,0])
    assert np.allclose(np.round(f._fit_df["std"],1),[1,1])
    assert np.allclose(np.round(f._fit_df["low_95"],0),[-2,-2])
    assert np.allclose(np.round(f._fit_df["high_95"],0),[2,2])


    # --------------------------------------------------------------------------
    # make sure the updater properly copies in parameter values the user may 
    # have altered after defining the model but before finalizing the fit. 

    def linear_fcn(m,b,x): return m*x + b
    x = np.linspace(-5,5,10)
    data_df = pd.DataFrame({"y_obs":linear_fcn(m=2,b=-1,x=x),
                            "y_std":0.1*np.ones(10)})

    f = BayesianSampler(some_function=linear_fcn,
                        non_fit_kwargs={"x":x})
    f.data_df = data_df

    # fit_df should have been populated with default values from param_df
    assert np.array_equal(f.fit_df["fixed"],[False,False])
    assert np.array_equal(f.fit_df["guess"],[0,0])
    assert np.array_equal(f.fit_df["prior_mean"],[np.nan,np.nan],equal_nan=True)
    assert np.array_equal(f.fit_df["prior_std"],[np.nan,np.nan],equal_nan=True)
    assert np.array_equal(f.fit_df["lower_bound"],[-np.inf,-np.inf])
    assert np.array_equal(f.fit_df["upper_bound"],[np.inf,np.inf])

    # update param_df
    f.param_df.loc["b","fixed"] = True
    f.param_df.loc["b","guess"] = 1
    f.param_df.loc["b","prior_mean"] = 5
    f.param_df.loc["b","prior_std"] = 3
    f.param_df.loc["m","upper_bound"] = 10
    f.param_df.loc["m","lower_bound"] = -10

    # no change in fit_df yet
    assert np.array_equal(f.fit_df["fixed"],[False,False])
    assert np.array_equal(f.fit_df["guess"],[0,0])
    assert np.array_equal(f.fit_df["prior_mean"],[np.nan,np.nan],equal_nan=True)
    assert np.array_equal(f.fit_df["prior_std"],[np.nan,np.nan],equal_nan=True)
    assert np.array_equal(f.fit_df["lower_bound"],[-np.inf,-np.inf])
    assert np.array_equal(f.fit_df["upper_bound"],[np.inf,np.inf])

    # run containing fit function from base class; that sets fit_has_been_run to
    # true.
    f.fit(num_walkers=100,
          num_steps=100,
          use_ml_guess=False,
          max_convergence_cycles=10)
    assert f._fit_has_been_run is True

    # now fit_df should have been updated with guesses etc. 
    assert np.array_equal(f.fit_df["fixed"],[False,True])
    assert np.array_equal(f.fit_df["guess"],[0,1])
    assert np.array_equal(f.fit_df["prior_mean"],[np.nan,5],equal_nan=True)
    assert np.array_equal(f.fit_df["prior_std"],[np.nan,3],equal_nan=True)
    assert np.array_equal(f.fit_df["lower_bound"],[-10,-np.inf])
    assert np.array_equal(f.fit_df["upper_bound"],[10,np.inf])
    

    # --------------------------------------------------------------------------
    # make sure the function handles a tiny number of samples

    f = BayesianSampler(some_function=linear_fcn,
                        non_fit_kwargs={"x":x})
    f.data_df = data_df

    assert f.samples is None

    # run containing fit function from base class; that sets fit_has_been_run to
    # true.
    f.fit(num_walkers=100,
          num_steps=100,
          use_ml_guess=False,
          max_convergence_cycles=10)
    assert f._fit_has_been_run is True

    assert f.samples.shape[0] > 90
    assert f.samples.shape[1] == 2
    f._samples = f._samples[:5,:]
    f._update_fit_df()

    with pytest.raises(TypeError):
        f.fit(num_walkers=10,
              num_steps=10,
              use_ml_guess=False,
              not_an_emcee_kwarg=5)


def test_BayesianSampler_fit_info():
    
    def test_fcn(m,b,x): return m*x + b
    f = BayesianSampler(some_function=test_fcn,
                        non_fit_kwargs={"x":np.arange(10)})
    y_obs = 2*np.arange(10) + 1
    y_std = 0.1

    assert len(f.fit_info) == 1
    assert f.fit_info["Final sample number"] is None

    f.fit(y_obs=y_obs,
          y_std=y_std,
          num_walkers=10,
          num_steps=100,
          burn_in=0.1,
          max_convergence_cycles=10)

    assert f.fit_info["Num walkers"] == f._num_walkers
    assert f.fit_info["Use ML guess"] == f._use_ml_guess
    assert f.fit_info["Num steps"] == f._num_steps
    assert f.fit_info["Burn in"] == f._burn_in
    assert f.fit_info["Num threads"] == f._num_threads
    assert f.fit_info["Max convergence cycles"] == 10

    # This will be some kind of big number after running to convergence
    assert f.fit_info["Final sample number"] > 100 
    f._samples = np.zeros((100,2))
    assert f.fit_info["Final sample number"] == 100
    

def test_BayesianSampler___repr__():
    
    # Stupidly simple fitting problem. find slope
    def model_to_wrap(m=1): return m*np.array([1,2,3])
    
    # Run _fit_has_been_run, success branch
    f = BayesianSampler(some_function=model_to_wrap)
    f.fit(y_obs=np.array([2,4,6]),
          y_std=[0.1,0.9,0.11],
          num_steps=100,
          max_convergence_cycles=10)

    out = f.__repr__().split("\n")
    assert len(out) == 24

    # hack, run _fit_has_been_run, _fit_failed branch
    f._success = False

    out = f.__repr__().split("\n")
    assert len(out) == 19 

    # Run not _fit_has_been_run
    f = BayesianSampler(some_function=model_to_wrap)
    
    out = f.__repr__().split("\n")
    assert len(out) == 9

