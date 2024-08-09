
import pytest

import numpy as np
import pandas as pd
from scipy import stats

from dataprob.fitters.bayesian import BayesianSampler
from dataprob.fitters.bayesian import _find_normalization
from dataprob.fitters.bayesian import _reconcile_bounds_and_priors
from dataprob.fitters.bayesian import _find_uniform_value

from dataprob.model_wrapper.model_wrapper import ModelWrapper


def test__find_normalization():
    
    # Calculate cdf and pdf over different interval (and in slightly different
    # way). Make sure the log_pdf is within 1% of log_cdf ground truth. 
    for scale in np.power(10.0,np.arange(-5,6)):

        frozen_rv = stats.norm(loc=0,scale=scale)
        res = np.finfo(frozen_rv.pdf(0).dtype).resolution
        offset = _find_normalization(scale=scale,
                                     rv=stats.norm)

        cdf = frozen_rv.cdf(0) - frozen_rv.cdf(-500000*res)
        pdf = np.sum(frozen_rv.pdf(np.linspace(-500000*res,0,500001)))
        if cdf == 0 or pdf == 0:
            print(f"skipping {scale} b/c of 0s: cdf {cdf}, pdf: {pdf}")
            continue
        
        log_cdf = np.log(cdf)
        log_pdf = np.log(pdf) + offset

        pct_diff = np.abs((log_pdf - log_cdf)/log_cdf)
        
        print(f"Trying {scale}",offset,log_pdf,log_cdf,pct_diff)
        
        assert pct_diff < 0.01
    
    # Calculate cdf and pdf over different interval (and in slightly different
    # way). Make sure the log_pdf is within 1% of log_cdf ground truth. Move 
    # loc to make sure it still works
    loc = 3
    for scale in np.power(10.0,np.arange(-1,6)):

        frozen_rv = stats.norm(loc=loc,scale=scale)
        res = np.finfo(frozen_rv.pdf(loc).dtype).resolution
        offset = _find_normalization(scale=scale,
                                     rv=stats.norm)

        cdf = frozen_rv.cdf(0) - frozen_rv.cdf(-500000*res)
        pdf = np.sum(frozen_rv.pdf(np.linspace(-500000*res,0,500001)))
        if cdf == 0 or pdf == 0:
            print(f"skipping {scale} b/c of 0s: cdf {cdf}, pdf: {pdf}")
            continue

        log_cdf = np.log(cdf)
        log_pdf = np.log(pdf) + offset

        pct_diff = np.abs((log_pdf - log_cdf)/log_cdf)
        
        print(f"Trying {scale}",offset,log_pdf,log_cdf,pct_diff)
        
        assert pct_diff < 0.01

    # Calculate cdf and pdf over different interval (and in slightly different
    # way). Make sure the log_pdf is within 1% of log_cdf ground truth. Move 
    # loc to make sure it still works
    loc = -3
    for scale in np.power(10.0,np.arange(8)):

        frozen_rv = stats.norm(loc=loc,scale=scale)
        res = np.finfo(frozen_rv.pdf(loc).dtype).resolution
        offset = _find_normalization(scale=scale,
                                     rv=stats.norm)

        cdf = frozen_rv.cdf(0) - frozen_rv.cdf(-500000*res)
        pdf = np.sum(frozen_rv.pdf(np.linspace(-500000*res,0,500001)))
        if cdf == 0 or pdf == 0:
            print(f"skipping {scale} b/c of 0s: cdf {cdf}, pdf: {pdf}")
            continue

        log_cdf = np.log(cdf)
        log_pdf = np.log(pdf) + offset

        pct_diff = np.abs((log_pdf - log_cdf)/log_cdf)
        
        print(f"Trying {scale}",offset,log_pdf,log_cdf,pct_diff)
        
        assert pct_diff < 0.01


def test__reconcile_bounds_and_priors():

    frozen_rv = stats.norm(loc=0,scale=1)
    
    res = np.finfo(frozen_rv.pdf(0).dtype).resolution

    result =_reconcile_bounds_and_priors(bounds=None,frozen_rv=frozen_rv)
    assert result == 0

    same_bounds = [[-1,-1],[0,0],[1,1]]
    for bounds in same_bounds:
        with pytest.warns():
            v =_reconcile_bounds_and_priors(bounds=bounds,frozen_rv=frozen_rv)
        assert v == 0

    frozen_rv = stats.norm(loc=0,scale=0.001)
    left = frozen_rv.cdf(1)
    for i in range(1,50):

        # Find a divisor of res that is sufficient to make the cdf different
        right = frozen_rv.cdf(1 + res/i)
        if left == right:
            super_close_bounds = [1,1+res/i]
    
        # If the bounds are the same within numerical error, we can't do this 
        # test -- break out
        if super_close_bounds[0] == super_close_bounds[1]:
            break

        print(f"Testing {super_close_bounds}")

        # Make sure the function warns they are numerically identical and 
        # returns infinity
        with pytest.warns():
            v =_reconcile_bounds_and_priors(bounds=super_close_bounds,frozen_rv=frozen_rv)
        assert v == 0

        break

    frozen_rv = stats.norm(loc=0,scale=1)
    v =_reconcile_bounds_and_priors(bounds=[-1,1],frozen_rv=frozen_rv)
    expected = np.log(1/(1 - (frozen_rv.cdf(-1) + frozen_rv.sf(1))))
    
    frozen_rv = stats.norm(loc=0,scale=1)
    v =_reconcile_bounds_and_priors(bounds=[0,1],frozen_rv=frozen_rv)
    expected = np.log(1/(1 - (frozen_rv.cdf(0) + frozen_rv.sf(1))))

    frozen_rv = stats.norm(loc=0,scale=1)
    v =_reconcile_bounds_and_priors(bounds=[-1,0],frozen_rv=frozen_rv)
    expected = np.log(1/(1 - (frozen_rv.cdf(-1) + frozen_rv.sf(0))))

    frozen_rv = stats.norm(loc=0,scale=10)
    v =_reconcile_bounds_and_priors(bounds=[-1,0],frozen_rv=frozen_rv)
    expected = np.log(1/(1 - (frozen_rv.cdf(-1) + frozen_rv.sf(0))))

    frozen_rv = stats.norm(loc=-2,scale=10)
    v =_reconcile_bounds_and_priors(bounds=[-1,0],frozen_rv=frozen_rv)
    expected = np.log(1/(1 - (frozen_rv.cdf(-1) + frozen_rv.sf(0))))

    assert np.isclose(v,expected)
    
def test__find_uniform_value():
    
    finfo = np.finfo(np.ones(1,dtype=float)[0].dtype)
    log_resolution = np.log(finfo.resolution)
    log_max = np.log(finfo.max)
    max_value = finfo.max

    bounds = None
    expected_value = log_resolution - (np.log(2) + log_max)
    value = _find_uniform_value(bounds)
    assert np.isclose(value,expected_value)

    bounds = [-np.inf,np.inf]
    expected_value = log_resolution - (np.log(2) + log_max)
    value = _find_uniform_value(bounds)
    assert np.isclose(value,expected_value)

    bounds = np.array([0,1])
    expected_value = log_resolution - np.log((bounds[1] - bounds[0]))
    value = _find_uniform_value(bounds)
    assert np.isclose(value,expected_value)

    with pytest.warns():
        v = _find_uniform_value([1,1])
    assert v == 0.0

    bounds = [-np.inf,0]
    expected_value = log_resolution - (log_max)
    value = _find_uniform_value(bounds)
    assert np.isclose(value,expected_value)
    
    bounds = [-np.inf,-max_value/2]
    expected_value = log_resolution - (np.log(0.5) + log_max)
    value = _find_uniform_value(bounds)
    assert np.isclose(value,expected_value)

    bounds = [-np.inf,max_value/2]
    expected_value = log_resolution - (np.log(1.5) + log_max)
    value = _find_uniform_value(bounds)
    assert np.isclose(value,expected_value)

    bounds = [0,np.inf]
    expected_value = log_resolution - (log_max)
    value = _find_uniform_value(bounds)
    assert np.isclose(value,expected_value)

    bounds = [max_value/2,np.inf]
    expected_value = log_resolution - (np.log(0.5) + log_max)
    value = _find_uniform_value(bounds)
    assert np.isclose(value,expected_value)

    bounds = [-max_value/2,np.inf]
    expected_value = log_resolution - (np.log(1.5) + log_max)
    value = _find_uniform_value(bounds)
    assert np.isclose(value,expected_value)


def test_BayesianSampler__init__():

    # default args work. check to make sure super().__init__ actually ran.
    f = BayesianSampler()
    assert f.fit_type == "bayesian"
    assert f._num_obs is None

    # args are being set
    f = BayesianSampler(num_walkers=100,
                       initial_walker_spread=1e-4,
                       ml_guess=True,
                       num_steps=100,
                       burn_in=0.1,
                       num_threads=1)

    assert f._num_walkers == 100
    assert np.isclose(f._initial_walker_spread,1e-4)
    assert f._ml_guess is True
    assert f._num_steps == 100
    assert np.isclose(f._burn_in,0.1)
    assert f._num_threads == 1
    assert f._success is None
    assert f.fit_type == "bayesian"

    # check num threads passing
    with pytest.raises(NotImplementedError):
        f = BayesianSampler(num_walkers=100,
                        initial_walker_spread=1e-4,
                        ml_guess=True,
                        num_steps=100,
                        burn_in=0.1,
                        num_threads=0)
        
    with pytest.raises(NotImplementedError):
        f = BayesianSampler(num_walkers=100,
                        initial_walker_spread=1e-4,
                        ml_guess=True,
                        num_steps=100,
                        burn_in=0.1,
                        num_threads=10)
    
    # Pass bad value into each kwarg to make sure checker is running
    with pytest.raises(ValueError):
        f = BayesianSampler(num_walkers=0)        
    with pytest.raises(ValueError):
        f = BayesianSampler(initial_walker_spread="stupid")
    with pytest.raises(ValueError):
        f = BayesianSampler(ml_guess="not a bool")
    with pytest.raises(ValueError):
        f = BayesianSampler(num_steps=1.2)
    with pytest.raises(ValueError):
        f = BayesianSampler(burn_in=0.0)
    with pytest.raises(ValueError):
        f = BayesianSampler(num_threads=-2)

def test__setup_priors():

    # basic functionality with a uniform and gaussian prior
    f = BayesianSampler()
    assert not hasattr(f,"_prior_frozen_rv")
    assert not hasattr(f,"_uniform_priors")
    assert not hasattr(f,"_gauss_prior_means")
    assert not hasattr(f,"_gauss_prior_stds")
    assert not hasattr(f,"_gauss_prior_offsets")
    assert not hasattr(f,"_gauss_prior_mask")

    f.priors = np.array([[0,np.nan],[1,np.nan]])
    f.bounds = np.array([[-np.inf,-np.inf],[np.inf,np.inf]])
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

    # No gaussian priors
    f = BayesianSampler()
    f.priors = np.array([[np.nan,np.nan],[np.nan,np.nan]])
    f.bounds = np.array([[-np.inf,-np.inf],[np.inf,np.inf]])
    f._setup_priors()

    assert len(f._gauss_prior_means) == 0
    assert len(f._gauss_prior_stds) == 0
    assert len(f._gauss_prior_offsets) == 0
    assert np.array_equal(f._gauss_prior_mask,[False,False])

    # No uniform priors
    f = BayesianSampler()
    f.priors = np.array([[1,2],[3,4]])
    f.bounds = np.array([[-np.inf,-np.inf],[np.inf,np.inf]])
    f._setup_priors()

    assert np.isclose(f._uniform_priors,0)
    assert np.allclose(f._gauss_prior_means,[1,2])
    assert np.allclose(f._gauss_prior_stds,[3,4])
    assert len(f._gauss_prior_offsets) == 2
    assert np.sum(f._gauss_prior_offsets < 0) == 2
    assert np.array_equal(f._gauss_prior_mask,[True,True])

    # check internal bounds calculation adjustment calculation
    f = BayesianSampler()
    f.priors = np.array([[10],[5]])
    f.bounds = np.array([[0],[20]])
    f._setup_priors()

    # Check parsing/basic run
    assert np.isclose(f._uniform_priors,0)
    assert np.array_equal(f._gauss_prior_means,[10])
    assert np.array_equal(f._gauss_prior_stds,[5])
    assert np.array_equal(f._gauss_prior_mask,[True])
    
    # Make sure final offset from the code matches what we calculate here. (Not
    # really testing math bit -- that's in the _find_normalization and 
    # _reconcile_bounds_and_priors tests).
    base_offset = _find_normalization(scale=1,rv=stats.norm)
    bounds = np.array([0,20])
    bounds_offset = _reconcile_bounds_and_priors(bounds=(bounds - 10)/5,
                                                 frozen_rv=stats.norm(loc=0,scale=1))
    assert np.isclose(base_offset + bounds_offset,f._gauss_prior_offsets[0])

def test_BayesianSampler_ln_prior():

    f = BayesianSampler()
    f.priors = np.array([[0],[1]])
    f.bounds = np.array([[-1],[1]])
    f._setup_priors()

    # Set up local calculation
    frozen_rv = stats.norm(loc=0,scale=1)
    base_offset = _find_normalization(scale=1,rv=stats.norm)
    bounds = np.array([-1,1])
    bounds_offset = _reconcile_bounds_and_priors(bounds=(bounds-0)/1,
                                                 frozen_rv=frozen_rv)
    

    # Try a set of values
    for v in [-0.9,-0.1,0.0,0.1,0.9]:

        print("testing",v)
        expected = frozen_rv.logpdf(v) + base_offset + bounds_offset
        value = f.ln_prior(param=np.array([v]))
        assert np.isclose(expected,value)

    # Go outside of bounds
    expected = -np.inf
    value = f.ln_prior(param=np.array([2]))
    assert np.isclose(expected,value)

    # Now set up two priors, one gauss with one infinite bound, one uniform with
    # infinte bound
    f = BayesianSampler()
    f.priors = np.array([[2,np.nan],[10,np.nan]])
    f.bounds = np.array([[-np.inf,-np.inf],[10,0]])
    f._setup_priors()

    # Set up local calculation
    frozen_rv = stats.norm(loc=0,scale=1)
    base_offset = _find_normalization(scale=1,rv=stats.norm)
    bounds = np.array([-np.inf,10])
    bounds_offset = _reconcile_bounds_and_priors(bounds=(bounds-2)/10,
                                                 frozen_rv=frozen_rv)
    
    total_gauss_offset = base_offset + bounds_offset
    uniform_prior = _find_uniform_value(bounds=np.array([-np.inf,0]))

    for v in [-1e6,-1,0,8]:
        print("testing",v)
        z = (v - 2)/10
        expected = frozen_rv.logpdf(z) + total_gauss_offset + uniform_prior
        value = f.ln_prior(np.array([v,-5]))
        assert np.isclose(expected,value)

    # Outside of bounds
    value = f.ln_prior(np.array([-1,2]))
    assert np.isclose(-np.inf,value)

def test_BayesianSampler__ln_prob(binding_curve_test_data):
    pass
    
    
def test_BayesianSampler_ln_prob(binding_curve_test_data):
    """
    Test calculation ,looking for proper error checking.
    """
    
    f = BayesianSampler()

    input_params = binding_curve_test_data["input_params"]

    # Should fail, haven't loaded a model, y_obs or y_stdev yet
    with pytest.raises(RuntimeError):
        f.ln_prob(input_params)

    f.model = binding_curve_test_data["generic_model"]

    # Should fail, haven't loaded y_obs or y_stdev yet
    with pytest.raises(RuntimeError):
        f.ln_prob(input_params)

    df = binding_curve_test_data["df"]
    f.y_obs = df.Y

    # Should fail, haven't loaded y_stdev yet
    with pytest.raises(RuntimeError):
        f.ln_prob(input_params)
    f.y_stdev = df.Y_stdev

    # Should fail, haven't loaded priors or bounds yet
    with pytest.raises(RuntimeError):
        f.ln_prob(input_params)
    
    # Set bounds and priors
    f.bounds = [[-np.inf],[np.inf]]
    f.priors = [[0],[2]]
    f._setup_priors()

    L = f.ln_prob(input_params)

def xtest_BayesianSampler__fit():
    pass

def xtest_BayesianSampler__update_estimates():
    pass

def xtest_BayesianSampler_fit_info():
    pass


@pytest.mark.slow
def test_fit(binding_curve_test_data,fit_tolerance_fixture):
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

def test_BayesianSampler___repr__():
    
    # Stupidly simple fitting problem. find slope
    def model_to_wrap(m=1,x=np.array([1,2,3])): return m*x
    mw = ModelWrapper(model_to_fit=model_to_wrap)

    # Run _fit_has_been_run, success branch
    f = BayesianSampler()
    f.model = mw
    f.fit(y_obs=np.array([2,4,6]),
          y_stdev=[0.1,0.1,0.1])

    out = f.__repr__().split("\n")
    assert len(out) == 23

    # hack, run _fit_has_been_run, _fit_failed branch
    f._success = False

    out = f.__repr__().split("\n")
    assert len(out) == 19    

    # Run not _fit_has_been_run
    f = BayesianSampler()
    f.model = mw

    out = f.__repr__().split("\n")
    assert len(out) == 15