
import pytest


from dataprob.fitters.bayesian._prior_processing import find_normalization
from dataprob.fitters.bayesian._prior_processing import reconcile_bounds_and_priors
from dataprob.fitters.bayesian._prior_processing import find_uniform_value
from dataprob.fitters.bayesian._prior_processing import _sample_gaussian
from dataprob.fitters.bayesian._prior_processing import _sample_uniform
from dataprob.fitters.bayesian._prior_processing import create_walkers


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
    pass

def test__sample_uniform():
    pass

def test_create_walkers():
    pass