import pytest

from dataprob.util.stats import durbin_watson
from dataprob.util.stats import ljung_box
from dataprob.util.stats import get_kde_max

import numpy as np

def test_durbin_watson():
    
    # perfect correlation
    r = np.ones(10)
    d, w = durbin_watson(residuals=r)
    assert w == "pos"
    assert np.isclose(d,0)

    # Random residuals
    r = np.array([-0.45472425, -0.17276604,  0.09131545,  0.41871986,
                  -0.64370331,  0.10271998, -0.34941661,  1.00628707,
                  -2.19315693,  0.05791554])
    d, w = durbin_watson(residuals=r)
    assert np.isclose(d,2.839380929949631)
    assert w == "ok"

    # change high_critical
    d, w = durbin_watson(residuals=r,high_critical=2.1)
    assert np.isclose(d,2.839380929949631)
    assert w == "neg"

    # negative correlation
    r = np.array([-1,1,-1,1,-1,1,-1,1,-1,1])
    d, w = durbin_watson(r)
    assert np.isclose(d,3.6)
    assert w == "neg"

    # positive correlation
    r = np.arange(10)
    d, w = durbin_watson(r)
    assert np.isclose(np.round(d,2),0.03)
    assert w == "pos"

    # Crank down low_critical
    d, w = durbin_watson(r,low_critical=0.01)
    assert np.isclose(np.round(d,2),0.03)
    assert w == "ok"

def test_ljung_box():

    # perfect
    r = np.ones(10)
    p, Q, df = ljung_box(residuals=r,num_param=0)
    assert np.isnan(p)
    assert np.isnan(Q)
    assert df == 0

    # random
    r = np.array([-0.45472425, -0.17276604,  0.09131545,  0.41871986,
                  -0.64370331,  0.10271998, -0.34941661,  1.00628707,
                  -2.19315693,  0.05791554])
    p, Q, df = ljung_box(residuals=r,num_param=0)
    assert np.round(p,2) == 0.78
    assert np.round(Q,2) == 4.01
    assert df == 7

    # negative
    r = np.array([-1,1,-1,1,-1,1,-1,1,-1,1])
    p, Q, df = ljung_box(residuals=r,num_param=0)
    assert np.isclose(np.round(np.log(p),2),-14.47)
    assert np.isclose(Q,42)
    assert df == 7

    # positive
    r = np.arange(10)
    p, Q, df = ljung_box(residuals=r,num_param=0)
    assert np.isclose(np.round(p,4),0.0008)
    assert np.isclose(np.round(Q,1),24.7)
    assert df == 7

    # test fit_df inputs
    r = np.arange(10)
    p, Q, df = ljung_box(residuals=r,num_param=0)
    assert df == 7
    assert np.isclose(np.round(p,4),0.0008)
    assert np.isclose(np.round(Q,1),24.7)

    p, Q, df = ljung_box(residuals=r,num_param=1)
    assert df == 8
    assert np.isclose(np.round(p,4),0.0017)
    assert np.isclose(np.round(Q,1),24.7)

    p, Q, df = ljung_box(residuals=r,num_param=2)
    assert df == 9
    assert np.isclose(np.round(p,4),0.0033)
    assert np.isclose(np.round(Q,1),24.7)

    p, Q, df = ljung_box(residuals=r,num_param=3)
    assert df == 10
    assert np.isclose(np.round(p,4),0.0059)
    assert np.isclose(np.round(Q,1),24.7)

def test_get_kde_max():
    
    # means of 0 1 2
    samples = np.random.multivariate_normal(mean=[0,1,2],
                                            cov=np.eye(3,dtype=float),
                                            size=10000)

    params = get_kde_max(samples=samples)
    assert np.array_equal(np.round(params,0),[0,1,2])

    # means of 5 0
    samples = np.random.multivariate_normal(mean=[5,0],
                                            cov=np.eye(2,dtype=float),
                                            size=10000)

    params = get_kde_max(samples=samples)
    assert np.array_equal(np.round(params,0),[5,0])


    # means of -1 2 8 12 8
    samples = np.random.multivariate_normal(mean=[-1,2,8,12,8],
                                            cov=np.eye(5,dtype=float),
                                            size=10000)

    params = get_kde_max(samples=samples)
    assert np.array_equal(np.round(params,0),[-1,2,8,12,8])


    # means of 0 1 2; make so every row has at least one nan. should revert to
    # np.nanmean
    samples = np.random.multivariate_normal(mean=[0,1,2],
                                            cov=np.eye(3,dtype=float),
                                            size=10000)
    samples[:,0] = np.nan

    with pytest.warns():
        params = get_kde_max(samples=samples)

    assert np.array_equal(np.round(params,0),[np.nan,1,2],equal_nan=True)


    
