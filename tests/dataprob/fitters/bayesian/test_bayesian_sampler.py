
import pytest

import numpy as np
import pandas as pd
from scipy import stats
import emcee

from dataprob.fitters.bayesian.bayesian_sampler import BayesianSampler
from dataprob.fitters.bayesian._prior_processing import find_normalization
from dataprob.fitters.bayesian._prior_processing import reconcile_bounds_and_priors
from dataprob.fitters.bayesian._prior_processing import find_uniform_value

from dataprob.model_wrapper.model_wrapper import ModelWrapper


def test_BayesianSampler__init__():

    # default args work. check to make sure super().__init__ actually ran.
    f = BayesianSampler()
    assert f.fit_type == "bayesian"
    assert f.num_obs is None

    # args are being set
    f = BayesianSampler(num_walkers=100,
                       use_ml_guess=True,
                       num_steps=100,
                       burn_in=0.1,
                       num_threads=1)

    assert f._num_walkers == 100
    assert f._use_ml_guess is True
    assert f._num_steps == 100
    assert np.isclose(f._burn_in,0.1)
    assert f._num_threads == 1
    assert f._success is None
    assert f.fit_type == "bayesian"

    # check num threads passing
    with pytest.raises(NotImplementedError):
        f = BayesianSampler(num_walkers=100,
                            use_ml_guess=True,
                            num_steps=100,
                            burn_in=0.1,
                            num_threads=0)
        
    with pytest.raises(NotImplementedError):
        f = BayesianSampler(num_walkers=100,
                            use_ml_guess=True,
                            num_steps=100,
                            burn_in=0.1,
                            num_threads=10)
    
    # Pass bad value into each kwarg to make sure checker is running
    with pytest.raises(ValueError):
        f = BayesianSampler(num_walkers=0)        
    with pytest.raises(ValueError):
        f = BayesianSampler(use_ml_guess="not a bool")
    with pytest.raises(ValueError):
        f = BayesianSampler(num_steps=1.2)
    with pytest.raises(ValueError):
        f = BayesianSampler(burn_in=0.0)
    with pytest.raises(ValueError):
        f = BayesianSampler(num_threads=-2)

def test__setup_priors():

    # Two parameter test function to wrap
    def test_fcn(a=1,b=2): return a*b

    # ----------------------------------------------------------------------
    # basic functionality with a uniform and gaussian prior

    f = BayesianSampler()
    assert not hasattr(f,"_prior_frozen_rv")
    assert not hasattr(f,"_uniform_priors")
    assert not hasattr(f,"_gauss_prior_means")
    assert not hasattr(f,"_gauss_prior_stds")
    assert not hasattr(f,"_gauss_prior_offsets")
    assert not hasattr(f,"_gauss_prior_mask")
    assert not hasattr(f,"_lower_bounds")
    assert not hasattr(f,"_upper_bounds")

    # Load model and set priors & bounds
    f.model = ModelWrapper(test_fcn)
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
    f = BayesianSampler()
    f.model = ModelWrapper(test_fcn)
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
    f = BayesianSampler()
    f.model = ModelWrapper(test_fcn)
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

    # Load model and set priors & bounds
    f = BayesianSampler()
    def four_param(a,b,c,d): return a*b*c*d
    f.model = ModelWrapper(four_param)
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
    f = BayesianSampler()
    def single_param(a): return a
    f.model = ModelWrapper(single_param)
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
    f = BayesianSampler()
    def single_param(a): return a
    f.model = ModelWrapper(single_param)
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

    # ----------------------------------------------------------------------
    # Now set up two priors, one gauss with one infinite bound, one uniform with
    # infinte bound

    f = BayesianSampler()
    def two_parameter(a=1,b=2): return a*b
    f.model = ModelWrapper(two_parameter)
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

def test_BayesianSampler_ln_prior(linear_fit):
    
    # test error checking. __ln_prior test checks numerical results 
    
    df = linear_fit["df"]
    fcn = linear_fit["fcn"]  # def simple_linear(m,b,x): return m*x + b
    linear_mw = ModelWrapper(fcn,fittable_params=["m","b"])
    linear_mw.x = df.x
    coeff = linear_fit["coeff"]
    param = np.array([coeff["m"],coeff["b"]])

    f = BayesianSampler()
    
    # should not work -- no model loaded
    with pytest.raises(RuntimeError):
        f.ln_prior(param)

    f.model = linear_mw
    v = f.ln_prior(param)
    assert v < 0

    with pytest.raises(ValueError):
        f.ln_prior(["a","b"])

    with pytest.raises(ValueError):
        f.ln_prior([1])

    with pytest.raises(ValueError):
        f.ln_prior([1,2,3])


def test_BayesianSampler__ln_prob(linear_fit):

    # Not really a numeric test, but makes sure the code is in fact summing 
    # ln_prior and ln_like and recognizing nan/inf correctly
    
    df = linear_fit["df"]
    fcn = linear_fit["fcn"]  # def simple_linear(m,b,x): return m*x + b
    linear_mw = ModelWrapper(fcn,fittable_params=["m","b"])
    linear_mw.x = df.x
    coeff = linear_fit["coeff"]
    param = np.array([coeff["m"],coeff["b"]])

    f = BayesianSampler()
    f.model = linear_mw
    f.y_obs = df.y_obs
    f.y_std = 0.1

    f.param_df["guess"] = param
    f.param_df["prior_mean"] = [2,1]
    f.param_df["prior_std"] = [2,2]
    f.param_df["lower_bound"] = [-np.inf,-np.inf]
    f.param_df["upper_bound"] = [np.inf,np.inf]
    f._model.finalize_params()
    f._setup_priors()

    ln_like = f.ln_like(param)
    ln_prob = f.ln_prior(param)

    assert f._ln_prob(param) == ln_like + ln_prob
    assert np.isinf(f._ln_prob(np.array([np.nan,1])))
    
def test_BayesianSampler_ln_prob(linear_fit):
    
    # test error checking. __ln_prob test checks numerical results 
    
    df = linear_fit["df"]
    fcn = linear_fit["fcn"]  # def simple_linear(m,b,x): return m*x + b
    linear_mw = ModelWrapper(fcn,fittable_params=["m","b"])
    linear_mw.x = df.x
    coeff = linear_fit["coeff"]
    param = np.array([coeff["m"],coeff["b"]])

    f = BayesianSampler()
    
    # should not work -- no model, y_obs, y_std loaded
    with pytest.raises(RuntimeError):
        f.ln_prob(param)

    # should not work -- no y_obs, y_std loaded
    f.model = linear_mw
    with pytest.raises(RuntimeError):
        f.ln_prob(param)
    
    # should not work -- no y_std loaded
    f.y_obs = df.y_obs
    with pytest.raises(RuntimeError):
        f.ln_prob(param)

    # should work -- all needed features defined
    f.y_std = df.y_std
    v = f.ln_prob(param)
    assert v < 0
    
    with pytest.raises(ValueError):
        f.ln_prob(["a","b"])

    with pytest.raises(ValueError):
        f.ln_prob([1])

    with pytest.raises(ValueError):
        f.ln_prob([1,2,3])

def test_BayesianSampler__fit(linear_fit):
    
    df = linear_fit["df"]
    fcn = linear_fit["fcn"]  # def simple_linear(m,b,x): return m*x + b
    linear_mw = ModelWrapper(fcn,fittable_params=["m","b"])
    linear_mw.x = df.x

    # -------------------------------------------------------------------------
    # basic run; checking use_ml_guess = True effect

    # Very small analysis, starting from ML
    f = BayesianSampler(num_walkers=10,
                        use_ml_guess=True,
                        num_steps=10,
                        burn_in=0.1,
                        num_threads=1)
    linear_mw = ModelWrapper(fcn,fittable_params=["m","b"])
    linear_mw.x = df.x
    f.model = linear_mw
    f.y_obs = df.y_obs
    f.y_std = df.y_std

    assert np.array_equal(f.param_df["guess"],[0,0])
    assert not hasattr(f,"_initial_state")
    assert np.sum(np.isnan(f._fit_df["estimate"])) == 2
    assert f.fit_result is None
    assert f.samples is None

    # run containing fit function from base class; that sets fit_has_been_run to
    # true. Make sure containing function ran completely. 
    f.fit()
    assert f._fit_has_been_run is True

    # These outputs are determined within ._fit
    assert issubclass(type(f._fit_result),emcee.ensemble.EnsembleSampler)
    assert f._initial_state.shape == (10,2)
    assert np.array_equal(f.samples.shape,[90,2]) 
    assert f._lnprob.shape == (90,)
    assert f._success is True
    assert np.sum(np.isnan(f.fit_df["estimate"])) == 0

    # -------------------------------------------------------------------------
    # basic run; checking use_ml_guess = False effect

    # Very small analysis, starting from ML
    f = BayesianSampler(num_walkers=10,
                        use_ml_guess=False,
                        num_steps=10,
                        burn_in=0.1,
                        num_threads=1)
    linear_mw = ModelWrapper(fcn,fittable_params=["m","b"])
    linear_mw.x = df.x
    f.model = linear_mw
    f.y_obs = df.y_obs
    f.y_std = df.y_std

    # set guess to 1, 1. works, but differs from ML guess of 2, 1 and can thus
    # be distinguished below
    f.param_df["guess"] = [1,1]
    assert not hasattr(f,"_initial_state")
    assert np.sum(np.isnan(f._fit_df["estimate"])) == 2
    assert f.fit_result is None
    assert f.samples is None

    # run containing fit function from base class; that sets fit_has_been_run to
    # true. Make sure containing function ran completely. 
    f.fit()
    assert f._fit_has_been_run is True

    # look for non-ML guess
    assert issubclass(type(f._fit_result),emcee.ensemble.EnsembleSampler)
    assert f._initial_state.shape == (10,2)
    assert np.array_equal(f.samples.shape,[90,2]) 
    assert f._lnprob.shape == (90,)
    assert f._success is True
    assert np.sum(np.isnan(f.fit_df["estimate"])) == 0

    # -------------------------------------------------------------------------
    # basic run; checking effects of altered num_steps and num_walkers

    # Very small analysis, starting from ML
    f = BayesianSampler(num_walkers=9,
                        use_ml_guess=True,
                        num_steps=20,
                        burn_in=0.1,
                        num_threads=1)
    linear_mw = ModelWrapper(fcn,fittable_params=["m","b"])
    linear_mw.x = df.x
    f.model = linear_mw
    f.y_obs = df.y_obs
    f.y_std = df.y_std

    assert np.array_equal(f.param_df["guess"],[0,0])
    assert not hasattr(f,"_initial_state")
    assert np.sum(np.isnan(f._fit_df["estimate"])) == 2
    assert f.fit_result is None
    assert f.samples is None

    # run containing fit function from base class; that sets fit_has_been_run to
    # true. Make sure containing function ran completely. 
    f.fit()
    assert f._fit_has_been_run is True

    # These outputs are determined within ._fit
    assert issubclass(type(f._fit_result),emcee.ensemble.EnsembleSampler)
    assert f._initial_state.shape == (9,2)
    assert np.array_equal(f.samples.shape,[162,2]) 
    assert f._lnprob.shape == (162,)
    assert f._success is True
    assert np.sum(np.isnan(f.fit_df["estimate"])) == 0

    # -------------------------------------------------------------------------
    # basic run; altered burn in

    # Very small analysis, starting from ML
    f = BayesianSampler(num_walkers=10,
                        use_ml_guess=True,
                        num_steps=10,
                        burn_in=0.5,
                        num_threads=1)
    linear_mw = ModelWrapper(fcn,fittable_params=["m","b"])
    linear_mw.x = df.x
    f.model = linear_mw
    f.y_obs = df.y_obs
    f.y_std = df.y_std

    assert np.array_equal(f.param_df["guess"],[0,0])
    assert not hasattr(f,"_initial_state")
    assert np.sum(np.isnan(f._fit_df["estimate"])) == 2
    assert f.fit_result is None
    assert f.samples is None

    # run containing fit function from base class; that sets fit_has_been_run to
    # true. Make sure containing function ran completely. 
    f.fit()
    assert f._fit_has_been_run is True

    # These outputs are determined within ._fit
    assert issubclass(type(f._fit_result),emcee.ensemble.EnsembleSampler)
    assert f._initial_state.shape == (10,2)
    assert np.array_equal(f.samples.shape,[50,2]) 
    assert f._lnprob.shape == (50,)
    assert f._success is True
    assert np.sum(np.isnan(f.fit_df["estimate"])) == 0

    # -------------------------------------------------------------------------
    # basic run; fixed parameter; no ml

    # Very small analysis, starting from no ML
    f = BayesianSampler(num_walkers=10,
                        use_ml_guess=False,
                        num_steps=10,
                        burn_in=0.1,
                        num_threads=1)
    linear_mw = ModelWrapper(fcn,fittable_params=["m","b"])
    linear_mw.param_df.loc["b","fixed"] = True
    linear_mw.param_df.loc["m","prior_mean"] = 2
    linear_mw.param_df.loc["m","prior_std"] = 5
    linear_mw.x = df.x
    f.model = linear_mw
    f.y_obs = df.y_obs
    f.y_std = df.y_std

    assert np.array_equal(f.param_df["guess"],[0,0])
    assert np.array_equal(f.param_df["fixed"],[False,True]) # make sure fixed
    assert not hasattr(f,"_initial_state")
    assert np.sum(np.isnan(f._fit_df["estimate"])) == 2
    assert f.fit_result is None
    assert f.samples is None

    # run containing fit function from base class; that sets fit_has_been_run to
    # true. Make sure containing function ran completely. 
    f.fit()
    assert f._fit_has_been_run is True

    # These outputs are determined within ._fit
    assert issubclass(type(f._fit_result),emcee.ensemble.EnsembleSampler)
    assert f._initial_state.shape == (10,1)
    assert np.array_equal(f.samples.shape,[90,1]) 
    assert f._lnprob.shape == (90,)
    assert f._success is True
    assert np.sum(np.isnan(f.fit_df["estimate"])) == 0
    assert f.fit_df.loc["b","estimate"] == f.fit_df.loc["b","guess"]

    # -------------------------------------------------------------------------
    # basic run; fixed parameter; ml

    # Very small analysis, starting from ML
    f = BayesianSampler(num_walkers=10,
                        use_ml_guess=True,
                        num_steps=10,
                        burn_in=0.1,
                        num_threads=1)
    linear_mw = ModelWrapper(fcn,fittable_params=["m","b"])
    linear_mw.param_df.loc["b","fixed"] = True
    linear_mw.x = df.x
    f.model = linear_mw
    f.y_obs = df.y_obs
    f.y_std = df.y_std

    assert np.array_equal(f.param_df["guess"],[0,0])
    assert np.array_equal(f.param_df["fixed"],[False,True]) # make sure fixed
    assert not hasattr(f,"_initial_state")
    assert np.sum(np.isnan(f._fit_df["estimate"])) == 2
    assert f.fit_result is None
    assert f.samples is None

    # run containing fit function from base class; that sets fit_has_been_run to
    # true. Make sure containing function ran completely. 
    f.fit()
    assert f._fit_has_been_run is True

    # These outputs are determined within ._fit
    assert issubclass(type(f._fit_result),emcee.ensemble.EnsembleSampler)
    assert f._initial_state.shape == (10,1)
    assert np.array_equal(f.samples.shape,[90,1]) 
    assert f._lnprob.shape == (90,)
    assert f._success is True
    assert np.sum(np.isnan(f.fit_df["estimate"])) == 0
    assert f.fit_df.loc["b","estimate"] == f.fit_df.loc["b","guess"]

    # -------------------------------------------------------------------------
    # run twice in a row to check for sample appending

    # Very small analysis, starting from ML
    f = BayesianSampler(num_walkers=10,
                        use_ml_guess=True,
                        num_steps=10,
                        burn_in=0.1,
                        num_threads=1)
    linear_mw = ModelWrapper(fcn,fittable_params=["m","b"])
    linear_mw.x = df.x
    f.model = linear_mw
    f.y_obs = df.y_obs
    f.y_std = df.y_std

    assert np.array_equal(f.param_df["guess"],[0,0])
    assert not hasattr(f,"_initial_state")
    assert np.sum(np.isnan(f._fit_df["estimate"])) == 2
    assert f.fit_result is None
    assert f.samples is None

    # run containing fit function from base class; that sets fit_has_been_run to
    # true. Make sure containing function ran completely. 
    f.fit()
    assert f._fit_has_been_run is True

    # These outputs are determined within ._fit
    assert issubclass(type(f._fit_result),emcee.ensemble.EnsembleSampler)
    assert f._initial_state.shape == (10,2)
    assert np.array_equal(f.samples.shape,[90,2]) 
    assert f._lnprob.shape == (90,)
    assert f._success is True
    assert np.sum(np.isnan(f.fit_df["estimate"])) == 0

    # now run again
    f.fit()

    assert issubclass(type(f._fit_result),emcee.ensemble.EnsembleSampler)
    assert f._initial_state.shape == (10,2)
    assert np.array_equal(f.samples.shape,[180,2]) 
    assert f._lnprob.shape == (180,)
    assert f._success is True
    assert np.sum(np.isnan(f.fit_df["estimate"])) == 0


def test_BayesianSampler__update_fit_df(linear_fit):
    
    # Create a BayesianSampler with a model loaded (and _fit_df implicitly 
    # created)
    f = BayesianSampler()
    def test_fcn(a=1,b=2): return a*b
    f.model = ModelWrapper(test_fcn)

    # add some fake samples
    f._samples = np.random.normal(loc=0,scale=1,size=(10000,2))

    assert np.sum(np.isnan(f._fit_df["estimate"])) == 2
    assert np.sum(np.isnan(f._fit_df["std"])) == 2
    assert np.sum(np.isnan(f._fit_df["low_95"])) == 2
    assert np.sum(np.isnan(f._fit_df["high_95"])) == 2

    f._update_fit_df()

    # Make sure mean/std/95 calc is write given fake samples we stuffed in
    assert np.allclose(np.round(f._fit_df["estimate"],1),[0,0])
    assert np.allclose(np.round(f._fit_df["std"],1),[1,1])
    assert np.allclose(np.round(f._fit_df["low_95"],0),[-2,-2])
    assert np.allclose(np.round(f._fit_df["high_95"],0),[2,2])


    # --------------------------------------------------------------------------
    # make sure the updater properly copies in parameter values the user may 
    # have altered after defining the model but before finalizing the fit. 

    df = linear_fit["df"]
    fcn = linear_fit["fcn"]  # def simple_linear(m,b,x): return m*x + b
    linear_mw = ModelWrapper(fcn,fittable_params=["m","b"])
    linear_mw.x = df.x

    # super small sampler
    f = BayesianSampler(num_walkers=10,
                        num_steps=10,
                        use_ml_guess=False)
    f.model = linear_mw
    f.y_obs = df.y_obs
    f.y_std = df.y_std

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
    f.fit()
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

    df = linear_fit["df"]
    fcn = linear_fit["fcn"]  # def simple_linear(m,b,x): return m*x + b
    linear_mw = ModelWrapper(fcn,fittable_params=["m","b"])
    linear_mw.x = df.x

    # super small sampler
    f = BayesianSampler(num_walkers=10,
                        num_steps=10,
                        use_ml_guess=False)
    f.model = linear_mw
    f.y_obs = df.y_obs
    f.y_std = df.y_std

    assert f.samples is None

    # run containing fit function from base class; that sets fit_has_been_run to
    # true.
    f.fit()
    assert f._fit_has_been_run is True

    assert f.samples.shape == (90,2)
    f._samples = f._samples[:5,:]
    f._update_fit_df()



def test_BayesianSampler_fit_info():
    
    f = BayesianSampler()
    assert f.fit_info["Num walkers"] == f._num_walkers
    assert f.fit_info["Use ML guess"] == f._use_ml_guess
    assert f.fit_info["Num steps"] == f._num_steps
    assert f.fit_info["Burn in"] == f._burn_in
    assert f.fit_info["Num threads"] == f._num_threads

    assert f.fit_info["Final sample number"] is None
    f._samples = np.zeros((100,2))
    assert f.fit_info["Final sample number"] == 100
    

def test_BayesianSampler___repr__():
    
    # Stupidly simple fitting problem. find slope
    def model_to_wrap(m=1,x=np.array([1,2,3])): return m*x
    mw = ModelWrapper(model_to_fit=model_to_wrap)

    # Run _fit_has_been_run, success branch
    f = BayesianSampler(num_steps=10)
    f.model = mw
    f.fit(y_obs=np.array([2,4,6]),
          y_std=[0.1,0.1,0.1])

    out = f.__repr__().split("\n")
    assert len(out) == 23

    # hack, run _fit_has_been_run, _fit_failed branch
    f._success = False

    out = f.__repr__().split("\n")
    assert len(out) == 18 

    # Run not _fit_has_been_run
    f = BayesianSampler(num_steps=10)
    f.model = mw

    out = f.__repr__().split("\n")
    assert len(out) == 14


@pytest.mark.slow
def xtest_fit(binding_curve_test_data,fit_tolerance_fixture):
    """
    Test the ability to fit the test data in binding_curve_test_data.
    """

    # Do fit using a generic unwrapped model and then creating and using a
    # ModelWrapper model instance

    for model_key in ["generic_model","wrappable_model"]:

        f = BayesianSampler()
        model = binding_curve_test_data[model_key]
        guesses = binding_curve_test_data["guesses"]
        df = binding_curve_test_data["df"]
        input_params = np.array(binding_curve_test_data["input_params"])

        if model_key == "wrappable_model":
            model = ModelWrapper(model)
            model.df = df
            model.K.bounds = [0,10]
        else:
            f.bounds = [[0],[10]]

        f.fit(model=model,guesses=guesses,y_obs=df.Y,y_stdev=df.Y_stdev)

        # Assert that we succesfully passed in bounds
        assert np.allclose(f.bounds,np.array([[0],[10]]))

        # Make sure fit worked
        assert f.success

        # Make sure fit gave right answer
        assert np.allclose(f.estimate,
                           input_params,
                           rtol=fit_tolerance_fixture,
                           atol=fit_tolerance_fixture*input_params)

        # Make sure mean of sampled uncertainty gives right answer
        sampled = np.mean(f.samples,axis=0)
        assert np.allclose(f.estimate,
                           sampled,
                           rtol=fit_tolerance_fixture,
                           atol=fit_tolerance_fixture*input_params)

        # Make sure corner plot call works and generates a plot
        corner_fig = f.corner_plot()
        assert corner_fig is not None

        # Make sure data frame that comes out is correct
        df = f.fit_df

        assert isinstance(df,pd.DataFrame)
        assert np.allclose(df["estimate"].iloc[:],
                           input_params,
                           rtol=fit_tolerance_fixture,
                           atol=fit_tolerance_fixture*input_params)
        assert np.array_equal(df["param"],f.names)
        assert np.array_equal(df["estimate"],f.estimate)
        assert np.array_equal(df["stdev"],f.stdev)
        assert np.array_equal(df["low_95"],f.ninetyfive[0,:])
        assert np.array_equal(df["high_95"],f.ninetyfive[1,:])
        assert np.array_equal(df["guess"],f.guesses)
        assert np.array_equal(df["lower_bound"],f.bounds[0,:])
        assert np.array_equal(df["upper_bound"],f.bounds[1,:])
        assert np.array_equal(f.samples.shape,(9000,1))