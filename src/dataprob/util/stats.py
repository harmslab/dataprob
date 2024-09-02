
import numpy as np
from scipy import stats
from scipy import optimize

def durbin_watson(residuals,
                  low_critical=1,
                  high_critical=3):
    """
    Run a Durbin-Watson test for residual autocorrelation. 

    Parameters
    ----------
    residuals : numpy.ndarray 
        fit residuals as a numpy array
    low_critical : float, default=1
        residuals with a test statistic less than lower_critical will be
        flagged as having a likely positive autocorrelation
    high_critical : float, default=3
        residuals with a test statistic greater than than high_critical will be
        flag as having a likely negative autocorrelation
    
    Returns
    -------
    d : float
        Durbin-Watson test statistic
    w : str
        result of test (pos, neg, ok)
    
    Notes
    -----
    Implements the Durbin-Watson test, whose test-statistic is :math:`D`. 

    + if :math:`D` is 2, it there is no detected correlation
    + if :math:`D` is less than 2, it is evidence of positive autocorrelation
    + if :math:`D` is more than 2, it is evidence of negative autocorrelation 
    """
    
    num = np.sum((residuals[1:] - residuals[:-1])**2)
    den = np.sum(residuals**2)
    
    d = num/den

    if d < low_critical: 
        w = "pos"
    elif d > high_critical: 
        w = "neg"
    else: 
        w = "ok"

    return d, w

def ljung_box(residuals,num_param=0):
    """
    Run a Ljung-Box test for residual autocorrelation. 

    Parameters
    ----------
    residuals : numpy.ndarray 
        fit residuals as a numpy array
    num_param : int, default=0
        number of fit parameters

    Returns
    -------
    p : float
        p-value for rejecting the null hypothesis that there is no 
        autocorrelation in the residuals 
    Q : float
        Ljung-Box test statistic
    df : int
        degrees of freedom

    Notes
    -----
    If the test statistic is higher than chi^2 for a given alpha and number of
    degrees of freedom, we reject the null hypothesis of uncorrelated residuals. 
    """

    # de-mean the residuals
    residuals = (residuals - np.mean(residuals))

    # If all zero return nan
    if np.sum(np.isclose(residuals,0)) == len(residuals):
        return np.nan, np.nan, 0

    # pre-calculate the normalization factor for the autocorrelation 
    # and bring in front of sum
    n = len(residuals)
    prefactor = n*(n + 2)/(np.sum(residuals**2))**2
    
    # Go through lags (k = 2 -> n-1)
    Q = 0 
    total = 0
    for k in range(2,n-1):
    
        at_atk = np.sum(residuals[k:]*residuals[:-k])
        
        Q += (at_atk**2)/(n - k)
    
        total+= 1
    
    Q = prefactor*Q
    df = (n - 3) + num_param
    p = 1 - stats.chi2.cdf(Q,df)
    
    return p, Q, df

def chi2(residuals,num_param):
    """
    Chi^2 goodness-of-fit test. Return the p-value for rejecting the null
    hypothesis that the model does not describe the data.

    Parameters
    ----------
    residuals : numpy.ndarray 
        fit residuals as a numpy array
    num_param : int
        number of fit parameters
    
    Returns
    -------
    p : float
        p value for rejecting null hypothesis that model describes data
    """
    
    chi2 = np.sum(residuals**2)

    num_obs = len(residuals)
    df = num_obs - num_param - 1

    return 1 - stats.chi2.cdf(chi2,df)

def chi2_reduced(residuals,num_param):
    """
    Reduced chi^2 goodness-of-fit test. Return chi^2/(num_obs - num_param),
    which measures fit quality. ~1: good fit; > 1: poor fit or uncertainties on
    each points underestimated; < 1: overfitting or uncertainties on each point 
    overestimated. 

    Parameters
    ----------
    residuals : numpy.ndarray 
        fit residuals as a numpy array
    num_param : int
        number of fit parameters
    
    Returns
    -------
    reduced_chi2 : float
        chi^2/(num_prob - num_param)
    """
    
    chi2 = np.sum(residuals**2)
    num_obs = len(residuals)
    df = num_obs - num_param - 1

    return chi2/df


def get_kde_max(samples):
    """
    Use a kernel density estimator to find the parameter estimates with the
    highest probability given a set of samples from the distribution.
    
    Parameters
    ----------
    samples : numpy.ndarray
        samples array of samples with dimensions (num_samples,num_parameters)
    
    Returns
    -------
    best_params : numpy.ndarray
        parameter values with the highest probability
    """
    
    # Create a mask for parameter samples where all parameters are not nan
    good_mask = np.sum(np.isnan(samples),axis=1) == 0

    # If no samples are good, just take the nanmean of the samples nd hope for
    # the best
    if np.sum(good_mask) == 0:
        return np.nanmean(samples,axis=0)
    
    # Take only good samples
    samples = samples[good_mask,:]

    # Build a gaussian kernel density estimator from the samples
    kde = stats.gaussian_kde(samples.T)

    # Find the kde maximum (minimize -kde)
    def to_optimize(x): return -kde(x)
    fit_result = optimize.minimize(to_optimize,x0=np.mean(samples,axis=0))

    # Get fit params
    return fit_result.x.copy()


