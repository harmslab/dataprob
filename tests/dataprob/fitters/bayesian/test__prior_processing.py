
import pytest


from dataprob.fitters.bayesian._prior_processing import find_normalization
from dataprob.fitters.bayesian._prior_processing import reconcile_bounds_and_priors
from dataprob.fitters.bayesian._prior_processing import find_uniform_value
from dataprob.fitters.bayesian._prior_processing import _sample_gaussian
from dataprob.fitters.bayesian._prior_processing import _cover_uniform
from dataprob.fitters.bayesian._prior_processing import create_walkers
from dataprob.model_wrapper.model_wrapper import ModelWrapper

import numpy as np
from scipy import stats

def test__find_normalization():
    
    # Calculate cdf and pdf over different interval (and in slightly different
    # way). Make sure the log_pdf is within 1% of log_cdf ground truth. 
    for scale in np.power(10.0,np.arange(-5,6)):

        frozen_rv = stats.norm(loc=0,scale=scale)
        res = np.finfo(frozen_rv.pdf(0).dtype).resolution
        offset = find_normalization(scale=scale,
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
        offset = find_normalization(scale=scale,
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
        offset = find_normalization(scale=scale,
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

    result = reconcile_bounds_and_priors(bounds=None,frozen_rv=frozen_rv)
    assert result == 0

    same_bounds = [[-1,-1],[0,0],[1,1]]
    for bounds in same_bounds:
        with pytest.warns():
            v = reconcile_bounds_and_priors(bounds=bounds,frozen_rv=frozen_rv)
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
            v = reconcile_bounds_and_priors(bounds=super_close_bounds,
                                            frozen_rv=frozen_rv)
        assert v == 0

        break

    frozen_rv = stats.norm(loc=0,scale=1)
    v = reconcile_bounds_and_priors(bounds=[-1,1],frozen_rv=frozen_rv)
    expected = np.log(1/(1 - (frozen_rv.cdf(-1) + frozen_rv.sf(1))))
    
    frozen_rv = stats.norm(loc=0,scale=1)
    v = reconcile_bounds_and_priors(bounds=[0,1],frozen_rv=frozen_rv)
    expected = np.log(1/(1 - (frozen_rv.cdf(0) + frozen_rv.sf(1))))

    frozen_rv = stats.norm(loc=0,scale=1)
    v = reconcile_bounds_and_priors(bounds=[-1,0],frozen_rv=frozen_rv)
    expected = np.log(1/(1 - (frozen_rv.cdf(-1) + frozen_rv.sf(0))))

    frozen_rv = stats.norm(loc=0,scale=10)
    v = reconcile_bounds_and_priors(bounds=[-1,0],frozen_rv=frozen_rv)
    expected = np.log(1/(1 - (frozen_rv.cdf(-1) + frozen_rv.sf(0))))

    frozen_rv = stats.norm(loc=-2,scale=10)
    v = reconcile_bounds_and_priors(bounds=[-1,0],frozen_rv=frozen_rv)
    expected = np.log(1/(1 - (frozen_rv.cdf(-1) + frozen_rv.sf(0))))

    assert np.isclose(v,expected)
    
def test__find_uniform_value():
    
    finfo = np.finfo(np.ones(1,dtype=float)[0].dtype)
    log_resolution = np.log(finfo.resolution)
    log_max = np.log(finfo.max)
    max_value = finfo.max

    bounds = None
    expected_value = log_resolution - (np.log(2) + log_max)
    value = find_uniform_value(bounds)
    assert np.isclose(value,expected_value)

    bounds = [-np.inf,np.inf]
    expected_value = log_resolution - (np.log(2) + log_max)
    value = find_uniform_value(bounds)
    assert np.isclose(value,expected_value)

    bounds = np.array([0,1])
    expected_value = log_resolution - np.log((bounds[1] - bounds[0]))
    value = find_uniform_value(bounds)
    assert np.isclose(value,expected_value)

    with pytest.warns():
        v = find_uniform_value([1,1])
    assert v == 0.0

    bounds = [-np.inf,0]
    expected_value = log_resolution - (log_max)
    value = find_uniform_value(bounds)
    assert np.isclose(value,expected_value)
    
    bounds = [-np.inf,-max_value/2]
    expected_value = log_resolution - (np.log(0.5) + log_max)
    value = find_uniform_value(bounds)
    assert np.isclose(value,expected_value)

    bounds = [-np.inf,max_value/2]
    expected_value = log_resolution - (np.log(1.5) + log_max)
    value = find_uniform_value(bounds)
    assert np.isclose(value,expected_value)

    bounds = [0,np.inf]
    expected_value = log_resolution - (log_max)
    value = find_uniform_value(bounds)
    assert np.isclose(value,expected_value)

    bounds = [max_value/2,np.inf]
    expected_value = log_resolution - (np.log(0.5) + log_max)
    value = find_uniform_value(bounds)
    assert np.isclose(value,expected_value)

    bounds = [-max_value/2,np.inf]
    expected_value = log_resolution - (np.log(1.5) + log_max)
    value = find_uniform_value(bounds)
    assert np.isclose(value,expected_value)

def test__sample_gaussian():
    
    # Generate with mean and std
    gaussian_priors = _sample_gaussian(prior_mean=1,
                                       prior_std=2,
                                       lower_bound=-np.inf,
                                       upper_bound=np.inf,
                                       num_walkers=10000)
    m = np.round(np.mean(gaussian_priors),0)
    s = np.round(np.std(gaussian_priors),0)
    assert np.allclose([m,s],[1,2])
    assert gaussian_priors.shape == (10000,)

    # Move mean and std
    gaussian_priors = _sample_gaussian(prior_mean=-1,
                                       prior_std=1,
                                       lower_bound=-np.inf,
                                       upper_bound=np.inf,
                                       num_walkers=10000)
    m = np.round(np.mean(gaussian_priors),0)
    s = np.round(np.std(gaussian_priors),0)
    assert np.allclose([m,s],[-1,1])
    assert gaussian_priors.shape == (10000,)

    # Crunch with very tight lower and upper bound --> should return None
    gaussian_priors = _sample_gaussian(prior_mean=0,
                                       prior_std=1,
                                       lower_bound=-0.00001,
                                       upper_bound=0.00001,
                                       num_walkers=10)
    assert gaussian_priors is None


def test__cover_uniform():

    # ---------------------------------------------------------------------
    # should create four walkers with "simple" case because both are on the
    # same side of zero
    walkers = _cover_uniform(lower_bound=1e6,
                              upper_bound=1e9,
                              num_walkers=4,
                              infinity_proxy=1e9)
    
    s = np.exp(np.linspace(0,10,4))
    s = (s - np.min(s))/(np.max(s) - np.min(s))
    expected = s*(1e9 - 1e6) + 1e6
    walkers = np.sort(walkers)
    assert np.array_equal(expected,walkers)
    
    # ---------------------------------------------------------------------
    # should create four walkers with "simple" case because both are on the
    # same side of zero
    walkers = _cover_uniform(lower_bound=-1e9,
                              upper_bound=-1e6,
                              num_walkers=4,
                              infinity_proxy=1e9)
    
    s = np.exp(np.linspace(0,10,4))
    s = (s - np.min(s))/(np.max(s) - np.min(s))
    expected = s*(-1e6 - -1e9) + -1e9
    walkers = np.sort(walkers)
    assert np.array_equal(expected,walkers)
    
    # ---------------------------------------------------------------------
    # should create four walkers with "simple" case because both are on the
    # same side of zero. Should use 1e10 because of infinity proxy
    walkers = _cover_uniform(lower_bound=1e6,
                              upper_bound=np.inf,
                              num_walkers=4,
                              infinity_proxy=1e10)
    
    s = np.exp(np.linspace(0,10,4))
    s = (s - np.min(s))/(np.max(s) - np.min(s))
    expected = s*(1e10 - 1e6) + 1e6
    walkers = np.sort(walkers)
    assert np.array_equal(expected,walkers)
    
    # ---------------------------------------------------------------------
    # should create four walkers with "simple" case because both are on the
    # same side of zero. Should use 1e10 because of infinity proxy
    walkers = _cover_uniform(lower_bound=0,
                              upper_bound=np.inf,
                              num_walkers=4,
                              infinity_proxy=1e10)
    
    s = np.exp(np.linspace(0,10,4))
    s = (s - np.min(s))/(np.max(s) - np.min(s))
    expected = s*(1e10 - 0) + 0
    walkers = np.sort(walkers)
    assert np.array_equal(expected,walkers)

    # ---------------------------------------------------------------------
    # should create four walkers with "simple" case because both are on the
    # same side of zero. Should use 1e10 because of infinity proxy
    walkers = _cover_uniform(lower_bound=-np.inf,
                              upper_bound=0,
                              num_walkers=4,
                              infinity_proxy=1e10)
    
    s = np.exp(np.linspace(0,10,4))
    s = (s - np.min(s))/(np.max(s) - np.min(s))
    expected = s*(0 - -1e10) + -1e10
    walkers = np.sort(walkers)
    assert np.array_equal(expected,walkers)

    # ---------------------------------------------------------------------
    # should create single walker at mean position because only one walker
    # requested
    walkers = _cover_uniform(lower_bound=-1e9,
                              upper_bound=1e9,
                              num_walkers=1,
                              infinity_proxy=1e9)
    
    expected = [0]
    assert np.allclose(expected,walkers)

    # ---------------------------------------------------------------------
    # should create single walker at mean position because only one walker
    # requested
    walkers = _cover_uniform(lower_bound=1,
                              upper_bound=3,
                              num_walkers=1,
                              infinity_proxy=1e9)
    
    expected = [2]
    assert np.allclose(expected,walkers)

    # ---------------------------------------------------------------------
    # should create four walkers with "complicated" case because wee're on
    # opposite sides of zero
    walkers = _cover_uniform(lower_bound=-np.inf,
                              upper_bound=np.inf,
                              num_walkers=4,
                              infinity_proxy=1e10)
    
    # Two below
    s = np.exp(np.linspace(0,10,2))
    s = (s - np.min(s))/(np.max(s) - np.min(s))
    below = s*-1e10

    # Two above
    t = np.exp(np.linspace(0,10,2))
    t = (t - np.min(t))/(np.max(t) - np.min(t))
    above = t*1e10

    # Get expected
    expected = list(below)
    expected.extend(list(above))
    expected = np.array(expected)
    expected = np.sort(expected)

    walkers = np.sort(walkers)
    assert np.array_equal(expected,walkers)

    # ---------------------------------------------------------------------
    # should create four walkers with "complicated" case because wee're on
    # opposite sides of zero
    walkers = _cover_uniform(lower_bound=-1,
                              upper_bound=np.inf,
                              num_walkers=4,
                              infinity_proxy=1e10)
    
    # One below
    below = [-1]

    # Three above
    t = np.exp(np.linspace(0,10,3))
    t = (t - np.min(t))/(np.max(t) - np.min(t))
    above = t*1e10

    # Get expected
    expected = list(below)
    expected.extend(list(above))
    expected = np.array(expected)
    expected = np.sort(expected)

    walkers = np.sort(walkers)
    assert np.array_equal(expected,walkers)

    # ---------------------------------------------------------------------
    # should create four walkers with "complicated" case because wee're on
    # opposite sides of zero
    walkers = _cover_uniform(lower_bound=-np.inf,
                              upper_bound=1,
                              num_walkers=4,
                              infinity_proxy=1e10)
    
    # Three below
    s = np.exp(np.linspace(0,10,3))
    s = (s - np.min(s))/(np.max(s) - np.min(s))
    below = s*-1e10

    # One above
    above = [1]

    # Get expected
    expected = list(below)
    expected.extend(list(above))
    expected = np.array(expected)
    expected = np.sort(expected)

    walkers = np.sort(walkers)
    assert np.array_equal(expected,walkers)


    walkers = _cover_uniform(lower_bound=-1,
                             upper_bound=0,
                             num_walkers=1000,
                             infinity_proxy=1e9)

def test_create_walkers():
    
    # test function to use
    def test_fcn(a,b,c): return a*b*c

    # ---------------------------------------------------------------------
    # Create a lot of walkers with gaussian priors and make sure the sampling
    # occurred properly

    mw = ModelWrapper(test_fcn)
    mw.param_df.loc[:,"prior_mean"] = [1,10,100]
    mw.param_df.loc[:,"prior_std"] = [3,2,1]
    mw.finalize_params()

    walkers = create_walkers(param_df=mw.param_df,
                             num_walkers=1000)
    assert walkers.shape == (1000,3)
    assert np.isclose(np.round(np.mean(walkers[:,0]),0),1) 
    assert np.isclose(np.round(np.mean(walkers[:,1]),0),10) 
    assert np.isclose(np.round(np.mean(walkers[:,2]),0),100) 

    assert np.isclose(np.round(np.std(walkers[:,0]),0),3) 
    assert np.isclose(np.round(np.std(walkers[:,1]),0),2) 
    assert np.isclose(np.round(np.std(walkers[:,2]),0),1) 
    
    # ---------------------------------------------------------------------
    # fix a parameter and make sure it's ignored

    mw = ModelWrapper(test_fcn)
    mw.param_df.loc[:,"prior_mean"] = [1,10,100]
    mw.param_df.loc[:,"prior_std"] = [3,2,1]
    mw.param_df.loc["a","fixed"] = True
    mw.finalize_params()

    walkers = create_walkers(param_df=mw.param_df,
                             num_walkers=1000)
    assert walkers.shape == (1000,2)
    assert np.isclose(np.round(np.mean(walkers[:,0]),0),10) 
    assert np.isclose(np.round(np.mean(walkers[:,1]),0),100) 

    assert np.isclose(np.round(np.std(walkers[:,0]),0),2) 
    assert np.isclose(np.round(np.std(walkers[:,1]),0),1)  

    # ---------------------------------------------------------------------
    # uniform priors spanning 0

    mw = ModelWrapper(test_fcn)
    mw.param_df.loc[:,"lower_bound"] = [-1,-2,-3]
    mw.param_df.loc[:,"upper_bound"] = [1,2,3]
    mw.finalize_params()
    assert np.sum(np.isnan(mw.param_df["prior_mean"])) == 3
    assert np.sum(np.isnan(mw.param_df["prior_std"])) == 3

    walkers = create_walkers(param_df=mw.param_df,
                             num_walkers=1000)
    
    # make sure we're sampling a big whack of the space, but only that space
    assert np.min(walkers[:,0]) >= -1
    assert np.min(walkers[:,0]) < -0.9
    assert np.max(walkers[:,0]) <=  1
    assert np.max(walkers[:,0]) > 0.9

    assert np.min(walkers[:,1]) >= -2
    assert np.min(walkers[:,1]) < -1.9
    assert np.max(walkers[:,1]) <=  2
    assert np.max(walkers[:,1]) > 1.9

    assert np.min(walkers[:,2]) >= -3
    assert np.min(walkers[:,2]) < -2.9
    assert np.max(walkers[:,2]) <=  3
    assert np.max(walkers[:,2]) > 2.9

    # ---------------------------------------------------------------------
    # uniform priors below zero (important to check because slightly different
    # algorithm for uniform priors spanning zero vs. being on one side)
    mw = ModelWrapper(test_fcn)
    mw.param_df.loc[:,"lower_bound"] = [-1,-2,-3]
    mw.param_df.loc[:,"upper_bound"] = [0,0,0]
    mw.finalize_params()
    assert np.sum(np.isnan(mw.param_df["prior_mean"])) == 3
    assert np.sum(np.isnan(mw.param_df["prior_std"])) == 3

    walkers = create_walkers(param_df=mw.param_df,
                             num_walkers=1000)
    
    # make sure we're sampling a big whack of the space, but only that space
    assert np.min(walkers[:,0]) >=  -1
    assert np.min(walkers[:,0]) < -0.9
    assert np.max(walkers[:,0]) <=   0
    assert np.max(walkers[:,0]) > -0.1

    assert np.min(walkers[:,1]) >=  -2
    assert np.min(walkers[:,1]) < -1.9
    assert np.max(walkers[:,1]) <=   0
    assert np.max(walkers[:,1]) > -0.1

    assert np.min(walkers[:,2]) >=  -3
    assert np.min(walkers[:,2]) < -2.9
    assert np.max(walkers[:,2]) <=   0
    assert np.max(walkers[:,2]) > -0.1

    # ---------------------------------------------------------------------
    # uniform priors above zero (important to check because slightly different
    # algorithm for uniform priors spanning zero vs. being on one side)
    mw = ModelWrapper(test_fcn)
    mw.param_df.loc[:,"lower_bound"] = [0,0,0]
    mw.param_df.loc[:,"upper_bound"] = [1,2,3]
    mw.finalize_params()
    assert np.sum(np.isnan(mw.param_df["prior_mean"])) == 3
    assert np.sum(np.isnan(mw.param_df["prior_std"])) == 3

    walkers = create_walkers(param_df=mw.param_df,
                             num_walkers=1000)
    
    # make sure we're sampling a big whack of the space, but only that space
    assert np.min(walkers[:,0]) >= 0.0
    assert np.min(walkers[:,0]) <  0.1
    assert np.max(walkers[:,0]) <= 1.0
    assert np.max(walkers[:,0]) >  0.9

    assert np.min(walkers[:,1]) >= 0.0
    assert np.min(walkers[:,1]) <  0.1
    assert np.max(walkers[:,1]) <= 2.0
    assert np.max(walkers[:,1]) >  1.9

    assert np.min(walkers[:,2]) >= 0.0
    assert np.min(walkers[:,2]) <  0.1
    assert np.max(walkers[:,2]) <= 3.0
    assert np.max(walkers[:,2]) >  2.9


    # ---------------------------------------------------------------------
    # gaussian priors that have tiny bounds and thus become uniform

    mw = ModelWrapper(test_fcn)
    mw.param_df.loc[:,"prior_mean"] = 0
    mw.param_df.loc[:,"prior_std"] = 3
    mw.param_df.loc[:,"lower_bound"] = -0.0001
    mw.param_df.loc[:,"upper_bound"] =  0.0001
    mw.finalize_params()
    assert np.sum(np.isnan(mw.param_df["prior_mean"])) == 0
    assert np.sum(np.isnan(mw.param_df["prior_std"])) == 0

    walkers = create_walkers(param_df=mw.param_df,
                             num_walkers=1000)
    
    assert np.min(walkers[:,0]) >= -0.0001
    assert np.max(walkers[:,0]) <=  0.0001

    # ---------------------------------------------------------------------
    # mix of gaussian and uniform priors
    
    mw = ModelWrapper(test_fcn)
    mw.param_df.loc[:,"guess"] = [0,0,1.5]
    mw.param_df.loc[:,"prior_mean"] = [0,np.nan,np.nan]
    mw.param_df.loc[:,"prior_std"] = [1,np.nan,np.nan]
    mw.param_df.loc[:,"lower_bound"] = [-np.inf,-1,1]
    mw.param_df.loc[:,"upper_bound"] =  [np.inf,1,2]
    mw.finalize_params()
    assert np.sum(np.isnan(mw.param_df["prior_mean"])) == 2
    assert np.sum(np.isnan(mw.param_df["prior_std"])) == 2

    walkers = create_walkers(param_df=mw.param_df,
                             num_walkers=1000)
    
    assert np.isclose(np.round(np.mean(walkers[:,0]),0),0) 
    assert np.isclose(np.round(np.std(walkers[:,0]),0),1) 

    assert np.min(walkers[:,1]) >= -1.0
    assert np.min(walkers[:,1]) <  -0.9
    assert np.max(walkers[:,1]) <=  1.0
    assert np.max(walkers[:,1]) >   0.9

    assert np.min(walkers[:,2]) >= 1.0
    assert np.min(walkers[:,2]) <  1.1
    assert np.max(walkers[:,2]) <= 2.0
    assert np.max(walkers[:,2]) >  1.9





