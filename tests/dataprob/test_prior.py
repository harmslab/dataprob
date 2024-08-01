import pytest

from dataprob.prior import _find_normalization
from dataprob.prior import _process_bounds

import numpy as np
from scipy import stats

def test_StatsPrior___init__():
    pass

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


def test__process_bounds():

    frozen_rv = stats.norm(loc=0,scale=1)
    
    res = np.finfo(frozen_rv.pdf(0).dtype).resolution

    result = _process_bounds(bounds=None,frozen_rv=frozen_rv)
    assert result == 0

    bad_bounds = [[],[0],["a",1],"test"]
    for b in bad_bounds:
        with pytest.raises(ValueError):
            _process_bounds(bounds=b,frozen_rv=frozen_rv)

    bad_bounds = [5,-1]
    with pytest.raises(ValueError):
        _process_bounds(bounds=bad_bounds,frozen_rv=frozen_rv)

    same_bounds = [[-1,-1],[0,0],[1,1]]
    for bounds in same_bounds:
        with pytest.warns():
            v = _process_bounds(bounds=bounds,frozen_rv=frozen_rv)
        assert np.isinf(v)

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
            v = _process_bounds(bounds=super_close_bounds,frozen_rv=frozen_rv)
        assert np.isinf(v)

        break

    frozen_rv = stats.norm(loc=0,scale=1)
    v = _process_bounds(bounds=[-1,1],frozen_rv=frozen_rv)
    expected = np.log(1/(1 - (frozen_rv.cdf(-1) + frozen_rv.sf(1))))
    
    frozen_rv = stats.norm(loc=0,scale=1)
    v = _process_bounds(bounds=[0,1],frozen_rv=frozen_rv)
    expected = np.log(1/(1 - (frozen_rv.cdf(0) + frozen_rv.sf(1))))

    frozen_rv = stats.norm(loc=0,scale=1)
    v = _process_bounds(bounds=[-1,0],frozen_rv=frozen_rv)
    expected = np.log(1/(1 - (frozen_rv.cdf(-1) + frozen_rv.sf(0))))

    frozen_rv = stats.norm(loc=0,scale=10)
    v = _process_bounds(bounds=[-1,0],frozen_rv=frozen_rv)
    expected = np.log(1/(1 - (frozen_rv.cdf(-1) + frozen_rv.sf(0))))

    frozen_rv = stats.norm(loc=-2,scale=10)
    v = _process_bounds(bounds=[-1,0],frozen_rv=frozen_rv)
    expected = np.log(1/(1 - (frozen_rv.cdf(-1) + frozen_rv.sf(0))))

    assert np.isclose(v,expected)
    
    