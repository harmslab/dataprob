"""
Functions for calculating priors used in Bayesian MCMC sampling.
"""

import numpy as np

import warnings

def find_normalization(scale,rv,**kwargs):
    """
    This method finds an offset to add to the output of rv.logpdf that 
    makes the total area under the pdf curve 1.0. 

    Parameters
    ----------
    scale : float
        scale argument to scipy.stats.rv
    rv : rv_continuous
        object for calculating logpdf
    kwargs : dict
        kwargs are passed to rv_continuous to initialize

    Returns
    -------
    offset : float
        offset to add to frozen_rv.logpdf(x) to normalize pdf to 1.0 

    Notes
    -----

    Calculate the minimum difference between floats we can represent using our
    data type. This sets the bin width for a discrete approximation of the
    continuous distribution.

        res = np.finfo(dtype).resolution
    
    Calculate the sum of the un-normalized pdf over a range spanning zero. We
    use a range of -num_to_same*res -> num_to_sample*res, taking steps of res:

        frozen_rv = rv(loc=0,scale=scale,**kwargs)
        x = np.arange(-num_to_sample*res,
                      num_to_sample*(res + 1),
                      res)
        un_norm_prob = np.sum(frozen_rv.pdf(loc=x))

    Calculate the difference in the cdf between num_to_sample*res and 
    -num_to_sample*res. This is the normalized probability of the slice
    centered on zero we calculated above. 

        norm_prob = cdf(num_to_sample*res) - cdf(-num_to_sample*res). 

    The ratio of norm_prob and un_norm_prob is a scalar that converts from raw
    pdf calculations to normalized probabilities. 

        norm_prob_x = frozen_rv.pdf(x)*norm_prob/un_norm_prob

    Since we care about log versions of this, it becomes:

        log_norm_prob_x = frozen_rv.logpdf(x) + log(norm_prob) - log(un_norm_prob)

    We can define a pre-calculated offset;

        offset = log(norm_prob) - log(un_norm_prob) 

    So, finally:

        log_norm_prob_x = frozen_rv.logpdf(x) + offset
    """

    # Create frozen distribution located at 0.0
    centered_rv = rv(loc=0,scale=scale,**kwargs)

    # Get smallest float step (resolution) for this datatype on this
    # platform.
    res = np.finfo(centered_rv.cdf(0).dtype).resolution

    num_to_sample = int(np.round(1/res/1e9,0))

    # Calculate prob of -1000res -> 1000res using cdf
    cdf = centered_rv.cdf(num_to_sample*res) - centered_rv.cdf(-num_to_sample*res)

    # Calculate pdf over this interval with a step size of res
    pdf = np.sum(centered_rv.pdf(np.linspace(-num_to_sample*res,
                                             num_to_sample*res,
                                             2*num_to_sample + 1)))
    
    # This normalizes logpdf. It's the log version of:
    # norm_prob_x = rv.pdf(x)*cdf/pdf. 
    offset = np.log(cdf) - np.log(pdf)
    
    return offset

def reconcile_bounds_and_priors(bounds,frozen_rv):
    """
    Figure out how much bounds trim off priors and return amount to add to the 
    prior offset area to set to pdf area to 1.

    Parameters
    ----------
    bounds : list-like or None
        bounds applied to parameter
    
    Returns
    -------
    offset : float
        offset to add to np.logpdf(x) that accounts for the fact that bounds 
        may have trimmed some of the probability density and normalizes the 
        area of the pdf to 1.0. 
    """

    # bounds specified, no offset to area
    if bounds is None:
        return 0.0

    # Parse bounds list.
    left = float(bounds[0])
    right = float(bounds[1])
    
    # left and right the same. prob of this value is 1 (np.log(1) = 0)
    if left == right:
        w = f"left and right bounds {bounds} are identical\n"
        warnings.warn(w)
        return 0.0
    
    # Calculate the amount the bounds trim off the top and the bottom of the 
    # distribution. 
    left_trim = frozen_rv.cdf(left) 
    right_trim = frozen_rv.sf(right)

    # Figure out how much we need to scale the area up given we lost some
    # of the tails. 
    remaining = 1 - (left_trim + right_trim)

    # If remaining ends up zero, the left and right edges of the bounds
    # are identical within numerical error. prob of this value is 1 (np.log(1) = 0)
    if remaining == 0:
        w = f"left and right bounds {bounds} are numerically identical\n"
        warnings.warn(w)
        return 0.0

    # Return amount to scale the area by
    return np.log(1/remaining)

def find_uniform_value(bounds):
    """
    Find the log probability for a specific value between bounds to use in a 
    uniform prior. 

    Parameters
    ----------
    bounds : list-like, optional
        list like float of two values. function assumes these are non-nanf 
        floats where bounds[1] >= bounds[0]. If bounds is None, assume 
        infinite bounds. 
    
    The idea here is to find the maximum finite width the parameter can occupy
    (bound[1] - bound[0]) and then to divide that by the number of steps based 
    on our numerical resolution. If our resolution was 0.1 and we were between 
    0 and 1, there would be 11 values, so the uniform prior would be ln(1/11).
    The log prior ends up being: np.log(resolution/width)

    For finite bounds, this is:
    
        np.log(resolution) - np.log(upper - lower)

    For infinite bounds, this is:
        
        np.log(resolution) - (np.log(scalar) + log(max_positive_finite_float)).

    max_positive_finite_float is the largest number we can represent. scalar
    ranges (0,2]. It would be 2 for the bounds (-infinity,infinity), because 
    this covers a span 2*max_positive_finite_float. To avoid overflow, we 
    represent as logs (ln(2*max) --> ln(2) + ln(max)). Scalar values lower than
    2 represent smaller chunks of the finite number line. A value of 1 would 
    be half the number line (-infinity,0), (0,infinity). The value approaches
    zero for the bounds (-infinity, -infinity + resolution) and 
    (infinity - resolution,infinity). 
    """

    # float architecture information
    finfo = np.finfo(np.ones(1,dtype=float)[0].dtype)
    log_resolution = np.log(finfo.resolution)
    log_max_value = np.log(finfo.max)
    max_value = finfo.max

    # width is 2*max_value --> log(2) + log_max_value
    if bounds is None:
        return log_resolution - (np.log(2) + log_max_value)

    # Parse bounds list
    left = float(bounds[0])
    right = float(bounds[1])

    # left and right the same, probability is 1.0 (log(P) = 0) for this value
    if left == right:
        w = f"left and right bounds {bounds} are identical\n"
        warnings.warn(w)
        return 0.0
    
    # width is 2*max_value --> log(2) + log_max_value
    if np.isinf(left) and np.isinf(right):
        return log_resolution - (np.log(2) + log_max_value)

    # Figure out scalars (see docstring)
    if np.isinf(left):

        # exactly half the number line; scalar = 1
        if right == 0:
            return log_resolution - log_max_value
        
        # scalar is less than 1 if left and right are both to the left of zero,
        # more than 1 if right is above zero
        if right < 0:
            scalar = 1 - np.abs(right/max_value)
        else:
            scalar = 1 + np.abs(right/max_value)
        
        return log_resolution - (np.log(scalar) + log_max_value)
    
    if np.isinf(right):
        
        # exactly half the number line; scalar = 1
        if left == 0:
            return log_resolution - log_max_value
        
        # scalar is less than 1 if left and right are both to the right of zero,
        # more than 1 if left is below zero
        if left > 0:
            scalar = 1 - np.abs(left/max_value)
        else:
            scalar = 1 + np.abs(left/max_value)
        
        return log_resolution - (np.log(scalar) + log_max_value)

    # simple case with finite bounds. resolution/bound_width
    return np.log(finfo.resolution) - np.log((right - left))

def _sample_gaussian(prior_mean,
                     prior_std,
                     lower_bound,
                     upper_bound,
                     num_walkers):
    """
    Attempt to generate num_walkers samples from a gaussian with prior_mean
    and prior_std, subject to the constraint that all samples are between 
    lower and upper bounds. Either return an array of samples num_walkers 
    long or, if no samples are found, return None. 
    """
    
    # generate a huge number of possible priors 
    gaussian_priors = np.random.normal(loc=prior_mean,
                                       scale=prior_std,
                                       size=num_walkers*1000)
    
    # Grab only those priors that are within the bounds
    good_mask = np.logical_and(gaussian_priors > lower_bound,
                               gaussian_priors < upper_bound)
    good_priors = gaussian_priors[good_mask]
    
    # If we have enough good priors, keep them. If we have only a few
    # good priors, it means the bounds have sliced out a ridiculously
    # tiny chunk of the distribution. Approximate the walkers as 
    # a uniform sample from that distribution. 
    if len(good_priors) >= num_walkers:
        return good_priors[:num_walkers]

    return None

def _cover_uniform(lower_bound,
                   upper_bound,
                   num_walkers,
                   infinity_proxy,
                   span_base=10):
    """
    Create samples at even steps (logarithmic) between lower and upper bound.
    Returns a numpy array num_walkers long with sampled values between lower
    and upper bound. These are shuffled, but steps rather than purely random
    samples. span_base sets 
    """

    # Slice down infinite bounds to largish numbers
    if np.isinf(lower_bound): 
        lower_bound = -infinity_proxy
    if np.isinf(upper_bound):
        upper_bound = infinity_proxy

    # If only one walker, put at the mean of the bounds
    if num_walkers == 1:
        return [np.mean([lower_bound,upper_bound])]

    # If the upper and lower bounds have the same sign, make a uniform
    # span between them (log steps). For example, 1e-9 to 1e-6 with four
    # walkers would yield approximately 1e-9, 1e-8, 1e-7, 1e-6 (uses ln,
    # so not quite powers of 10)
    if upper_bound*lower_bound >= 0:
        
        steps = np.exp(np.linspace(0,span_base,num_walkers))
        steps = (steps - np.min(steps))/(np.max(steps) - np.min(steps))
        walkers = steps*(upper_bound - lower_bound) + lower_bound
        np.random.shuffle(walkers)
        
        return walkers

    # If the upper and lower bounds have different signs, make uniform
    # spans from 0 to upper and 0 to lower, weighted by how much of the
    # the span is above and below. 
    
    # Figure out fraction of uniform distribution below zero
    lower_mag = np.abs(lower_bound)
    upper_mag = np.abs(upper_bound)
    fx_lower = lower_mag/(lower_mag + upper_mag)

    # Figure out how many walkers to place above and below zero
    num_below = int(np.round(fx_lower*num_walkers,0))

    # Make sure we have at least one above and one below
    if num_below == 0: 
        num_below = 1
    if num_below == num_walkers:
        num_below = num_walkers - 1
    num_above = num_walkers - num_below

    # Create steps from 0 to upper_bound
    if num_above > 1:
        steps = np.exp(np.linspace(0,span_base,num_above))
        steps = (steps - np.min(steps))/(np.max(steps) - np.min(steps))
        above_walkers = list(steps*upper_bound)
    else:
        above_walkers = [upper_bound]

    # Create steps from 0 to lower_bound
    if num_below > 1:
        steps = np.exp(np.linspace(0,span_base,num_below))
        steps = (steps - np.min(steps))/(np.max(steps) - np.min(steps))
        below_walkers = list(steps*lower_bound)
    else:
        below_walkers = [lower_bound]

    # Combine all steps
    above_walkers.extend(below_walkers)
    walkers = np.array(above_walkers)

    # Shuffle randomly
    np.random.shuffle(walkers)

    return walkers

def create_walkers(param_df,
                   num_walkers,
                   infinity_proxy=1e9):
    """
    Create a collection of starting walkers from a parameter dataframe. 
    
    Parameters
    ----------
    param_df : pandas.DataFrame
        parameter dataframe (usually taken from a ModelWrapper instance) that
        should have fixed, guess, prior_mean, prior_std, lower_bound, and
        upper_bound columns. This dataframe is not validated by the function; 
        this is the callers responsibility.
    num_walkers : int
        number of walkers to generate. 
    infinity_proxy : float, default = 1e9
        substitute this for infinite bounds. should generally be a large-ish
        number, but not so large as to lead to numerical problems. 

    Returns
    -------
    walkers : numpy.ndarray
        numpy array with dimensions (num_walkers,num_parameters) with sampled
        starting points for an MCMC calculation 
    """

    walker_list = []

    # Go through each parameter one-by-one
    for p in param_df.index:
        
        # Skip fixed parameters
        if param_df.loc[p,"fixed"]:
            continue

        # Get prior mean, std, and bounds
        prior_mean = param_df.loc[p,"prior_mean"]
        prior_std = param_df.loc[p,"prior_std"]
        lower_bound = param_df.loc[p,"lower_bound"]
        upper_bound = param_df.loc[p,"upper_bound"]

        # If gaussian prior, try to do that first. 
        if not np.isnan(prior_mean):

            gaussian_priors = _sample_gaussian(prior_mean,
                                               prior_std,
                                               lower_bound,
                                               upper_bound,
                                               num_walkers)
            if gaussian_priors is not None:
                walker_list.append(gaussian_priors)
                continue

        # If we get here, gaussian priors were not given or did not work.
        uniform_priors = _cover_uniform(lower_bound,
                                        upper_bound,
                                        num_walkers,
                                        infinity_proxy)
        walker_list.append(uniform_priors)
        
    walkers = np.array(walker_list).T

    return walkers