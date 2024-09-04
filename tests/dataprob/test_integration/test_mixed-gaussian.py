"""
First a six-parameter mixed gaussian. Uses a vector first argument, 
demonstrating this capability. 
"""

import pytest

import dataprob

import numpy as np
import matplotlib

def _core_test(method,**method_kwargs):
    
    # ------------------------------------------------------------------------
    # Define model and generate data

    from scipy import stats

    def multi_gaussian(params,num_gaussians,x):
        """
        Generate a multi-guassian.

        Parameters
        ----------
        params : numpy.ndarray
            float numpy array that is num_gaussians*3 long. this encodes the
            gaussian [mean1,std1,area1,mean2,std2,area2,...meanN,stdN,areaN]
            shape parameters
        num_gaussians : int
            number of gaussians in the params array
        x : numpy.ndarray
            calculate guassians over the values in x 

        Returns
        -------
        out : numpy.ndarray
            sum of the pdfs for the gaussians in params calculated over x
        """

        # Create output array
        out = np.zeros(len(x),dtype=float)

        # For each gaussian
        for i in range(num_gaussians):

            # Grab the shape parameters
            mean = params[i*3]
            std = params[i*3 + 1]
            area = params[i*3 + 2]

            # Add this to out
            out += area*stats.norm(loc=mean,scale=std).pdf(x)

        return out
    
    gen_params = {"params":np.array([5,0.3,10,6,1.5,10]),
                "num_gaussians":2}

    err = 0.2
    num_points = 50

    x = np.linspace(0,10,num_points)
    y_obs = multi_gaussian(x=x,**gen_params) + np.random.normal(0,err,num_points)
    y_std = 2*err

    test_fcn = multi_gaussian
    non_fit_kwargs = {"x":x,
                    "num_gaussians":2}


    # ------------------------------------------------------------------------
    # Run analysis

    f = dataprob.setup(some_function=test_fcn,
                    method=method,
                    fit_parameters=["m0","s0","a0","m1","s1","a1"],
                    non_fit_kwargs=non_fit_kwargs,
                    vector_first_arg=True)

    # Set some guesses
    f.param_df.loc[["m0","s0","a0","m1","s1","a1"],"guess"] = [5,1,1,7,1,1]

    f.fit(y_obs=y_obs,
          y_std=y_std,
          **method_kwargs)

    # make sure estimate lands between confidence intervals
    expected = np.array(gen_params["params"])
    assert np.sum(expected < np.array(f.fit_df["low_95"])) == 0
    assert np.sum(expected > np.array(f.fit_df["high_95"])) == 0
    
    fig = dataprob.plot_summary(f)
    assert issubclass(type(fig),matplotlib.figure.Figure)
    matplotlib.pyplot.close(fig)

    fig = dataprob.plot_corner(f)
    assert issubclass(type(fig),matplotlib.figure.Figure)
    matplotlib.pyplot.close(fig)

# Try tests twice. We do a lot of tests around 95% confidence intervals. Odds
# are relatively high we hit one across the whole suite. So try once; if fails,
# try again. If it fails again, fail completely

def test_ml():
    
    try:
        _core_test(method="ml")
    except AssertionError:
        _core_test(method="ml")

@pytest.mark.slow
def test_bayesian():

    try:
        _core_test(method="mcmc",max_convergence_cycles=10)
    except AssertionError:
        _core_test(method="mcmc",max_convergence_cycles=10)

@pytest.mark.slow
def test_bootstrap():

    try:
        _core_test(method="bootstrap")
    except AssertionError:
        _core_test(method="bootstrap")