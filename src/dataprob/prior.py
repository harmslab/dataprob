


import numpy as np
from scipy import stats

import warnings

def _find_normalization(scale,rv,**kwargs):
    """
    This method finds an offset to add to the output of rv.logpdf that 
    makes the total area under the pdf curve 1.0. 

    First, calculate the unnormalized pdf at the center of the distribution
    (loc):

        un_norm_prob = rv.pdf(loc)

    Second, calculate the minimum difference between floats we can represent
    using our data type. This sets the bin width for a discrete approximation
    of the continuous distribution.

        res = np.finfo(un_norm_prob.dtype).resolution

    Third, calculate the difference in the cdf between loc + num_to_sample*res
    and loc - num_to_sample*res. This is the normalized probability of the slice
    centered on loc. 

        norm_prob = cdf(loc + num_to_sample*res) - cdf(loc - num_to_sample*res). 

    The ratio of norm_prob and un_norm_prob is a scalar that converts from raw
    pdf calculations to normalized probabilities. 

        norm_prob_x = rv.pdf(x)*norm_prob/un_norm_prob

    Since we care about log versions of this, it becomes:

        log_norm_prob_x = rv.logpdf(x) + log(norm_prob) - log(un_norm_prob)

    We can define a pre-calculated offset;

        offset = log(norm_prob) - log(un_norm_prob) 

    So, finally:

        log_norm_prob_x = rv.logpdf(x) + offset

    This is most numerically stable at loc = 0, so figure out the
    normalization using all other shape parameters but loc = 0. 
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

def _process_bounds(bounds,frozen_rv):

    # bounds specified, no offset to area
    if bounds is None:
        return 0.0

    # Parse bounds list
    try: 
        left = float(bounds[0])
        right = float(bounds[1])
    except Exception as e:
        err = "bounds must be an iterable with an upper and lower value\n"
        raise ValueError(err) from e

    # Check sanity of bounds
    if left > right:
        err = f"left bound '{bounds[0]}' must be smaller than right bound '{bounds[1]}'\n"
        raise ValueError(err)

    # left and right the same, infinite prior for the shared value
    if left == right:
        w = f"left and right bounds {bounds} are identical\n"
        warnings.warn(w)
        return np.inf
    
    # Calculate the amount the bounds trim off the top and the bottom of the 
    # distribution. 
    left_trim = frozen_rv.cdf(left) 
    right_trim = frozen_rv.sf(right)

    # Figure out how much we need to scale the area up given we lost some
    # of the tails. 
    remaining = 1 - (left_trim + right_trim)

    # If remaining ends up zero, the left and right edges of the bounds
    # are identical within numerical error
    if remaining == 0:
        w = f"left and right bounds {bounds} are numerically identical\n"
        warnings.warn(w)
        return np.inf

    # Return amount to scale the area by
    return np.log(1/remaining)
    

class StatsPrior:
    
    def __init__(self,
                 loc=0,
                 scale=1,
                 rv=stats.norm,
                 bounds=None,
                 **kwargs):

        if not issubclass(type(stats.norm),stats.rv_continuous):
            err = f"rv '{rv}' should be a scipy.stats.rv_continuous distribution\n"
            raise ValueError(err)
                
        self._frozen = rv(loc=loc,scale=scale,**kwargs)        
        self._offset = _find_normalization(scale,rv,**kwargs)
        self._offset += _process_bounds(bounds)
    
    def ln_prior(self,x):

        return self._frozen.logpdf(x) + self._offset

class UniformPrior:

    def __init__(self,bounds=None):

        
        finfo = np.finfo(np.ones(1)[0].dtype)
        res = finfo.resolution
        
        if bounds is None:
            self._value = np.log(res) - (np.log(2) + np.log(finfo.max))
        else:
            self._value = np.log(res) - np.log((bounds[1] - bounds[0]))

    def ln_prior(self,x):

        return self._value
        
        
