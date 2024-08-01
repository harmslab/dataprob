"""
Class for holding fit parameters, including guesses, values, ranges, etc.
"""

import numpy as np

import warnings

from dataprob.check import check_bool
from dataprob.check import check_float

_INFINITY_PROXY = 1e9

def _guess_setter_from_bounds(bounds):
    """
    Find reasonable guesses from bounds.

    Parameters
    ----------
    bounds : numpy.ndarray
        array with two entries corresponding to lower and upper bounds. because
        this is private, it assumes bounds[0] < bounds[1]. 

    Returns
    -------
    guess : float
        parameter guess consistent with these bounds
    """

    # Copy bounds because we will edit
    bounds = np.array(bounds,dtype=float).copy()

    # Approximate infinities as a moderately large, finite number
    if np.isinf(bounds[0]):
        bounds[0] = -_INFINITY_PROXY
    
    if np.isinf(bounds[1]):
        bounds[1] = _INFINITY_PROXY
    
    # Lower bound is 0, return upper_bound/2
    if bounds[0] == 0:
        guess = bounds[1]/2
        return guess
    
    # Upper bound is 0, return lower_bound/2
    if bounds[1] == 0:
        guess = bounds[0]/2
        return guess

    lower_sign = bounds[0]/np.abs(bounds[0])
    upper_sign = bounds[1]/np.abs(bounds[1])

    # If the bounds have the same sign, use geometric mean. Otherwise, 
    # use arithmetic mean
    if upper_sign == lower_sign:
        log_sum = np.sum(np.log(np.abs(bounds)))
        guess = upper_sign * np.exp(log_sum/2)
    else:
        guess = np.mean(bounds)

    return guess


class FitParameter:
    """
    Class for storing and manipulating fit parameters.

    Fit control attributes
    ----------------------

    These attributes can be set on __init__ or directly via instance.xxx = value.
    If any of these values are set, the fit result (stored in .value) is wiped
    out to avoid inconsistency between the fit control parameters and the
    resulting fit.

        + .guess: (float) guess value for fit
        + .fixed: (bool) whether to float parameter or not
        + .bounds: [float,float] lower and upper bounds for parameter
        + .prior: [float,float] mean and standard deviation of gaussian prior

    Fit result attributes
    ---------------------

    These attributes can only be updated using the load_fit_result method, which
    takes a fitter instance as an argument.

        + .value: (float) Current value of parameter.
                  Before the fit is run, this will be the .guess value. After
                  the fit is run, this will be the parameter estimate.
        + .stdev: (float) standard deviation on the estimate of the fit parameter.
                  Before the fit is run, this will be None.
        + .ninetyfive: (float,float) top and bottom of the 95% confidence interval.
                  Before the fit is run, this will be None.
        + .is_fit_result: (bool) whether or not the parameter contains the results
                          of a fit.

    Metadata attributes
    -------------------

    This attribute can be set on __init__ or directly via instance.name = value.
    The value is used for convenience only and is ignored by other methods.

        + .name: (str) name of fit parameter
    """

    def __init__(self,name,guess=None,fixed=False,bounds=None,prior=None):
        """
        Initialize class.  

        Parameters
        ----------
        name : str
            name of parameter
        guess : float, optional. 
            parameter guess. If None, the guess will be determined in the 
            following way.  1) If prior is given, the guess will be set to the
            mean of the prior. 2) If no prior is given and lower and upper
            bounds have the same sign, the guess will be placed at the geometric
            mean of the two bounds. 3) If no prior is given and lower and upper
            bounds have different signs, the guess is placed at the arithmetic
            mean of the two bounds. 4) If no prior or bounds are given, the
            guess will be set to 0.0.
        fixed : bool
            whether or not the parameter is fixed
        bounds : iterable
            bounds (inclusive) on fit for parameter (list-like, 2 floats). If
            not given, bounds will be set to (-np.inf,np.inf).  Nones are
            interpreted as infinities: (None,5) would give bounds of (-np.inf,5)
        prior : iterable
            prior on parameter (list-like, 2 floats). The two values represent
            the (mean,stdev) for a Gaussian prior on that parameter. The second
            value must be positive. Infinities are not allowed. If None,
            the prior will be set to (np.nan,np.nan), causing a Bayesian
            inference to use a uniform prior. 
        """

        # Setting must be in this order. If no guess is specified, the guess
        # is made first based on the prior, then (if not specified) based on
        # the bounds.
        self._is_fit_result = False
        self.bounds = bounds
        self.prior = prior
        self.guess = guess

        # Simple terms
        self.name = name
        self.fixed = fixed

    #--------------------------------------------------------------------------
    # parameter name

    @property
    def name(self):
        """
        Name of the parameter.
        """

        try:
            return self._name
        except AttributeError:
            return None

    @name.setter
    def name(self,name):
        self._name = str(name)


    #--------------------------------------------------------------------------
    # parameter guess

    @property
    def guess(self):
        """
        Guess for the parameter.
        """

        try:
            return self._guess
        except AttributeError:
            return None

    @guess.setter
    def guess(self,guess):
        f"""
        Set the guess.  If None, the guess is assigned using the following. 1)
        If prior is specified, the mean of the prior is used as the guess. 2) 
        If bounds are specified and both have the same sign, the guess becomes
        the to arithmetic mean of the bounds. 3) If the bounds are specified and
        have different signs, use the geometric mean of the bounds. (For #2 and
        #3, if one of the bounds is infinite, set that bound to {_INFINITY_PROXY} 
        for the purpose of the mean calculation. 4) If no prior or bound is 
        specified, the guess is set to 0.0. 
        """

        if guess is None:
            if not np.isnan(self.prior[0]):
                guess = self.prior[0]
            else:
                guess = _guess_setter_from_bounds(self.bounds)

        guess = check_float(value=guess,
                            variable_name="guess")



        # Make sure the guess is within bounds
        if guess > self.bounds[1] or guess < self.bounds[0]:
            err = f"guess ({guess}) outside bounds ({self.bounds[0],self.bounds[1]})\n"
            raise ValueError(err)

        self._guess = guess
        self._clear_fit_result()

    #--------------------------------------------------------------------------
    # parameter fixed-ness.

    @property
    def fixed(self):
        """
        Whether or not the parameter if fixed.
        """

        return self._fixed

    @fixed.setter
    def fixed(self,bool_value):
        """
        Fix or unfix the parameter.
        """

        self._fixed = check_bool(value=bool_value,
                                 variable_name="fixed")
        self._clear_fit_result()

    #--------------------------------------------------------------------------
    # bounds for fit.

    @property
    def bounds(self):
        """
        Fit bounds.  Either list of bounds or None.
        """

        try:
            return self._bounds
        except AttributeError:
            return None

    @bounds.setter
    def bounds(self,bounds):
        """
        Set fit bounds.
        """

        err_msg = \
        """

        Bounds should be list-like, with two floats. The first entry is the 
        lower bound; the second is the upper bounds. When doing the fit, bounds
        are inclusive. 
        
        + The upper bound must be larger than the lower bound. 
        + If bounds == None, the bounds are set to [-np.inf,np.inf]. 
        + np.inf values are allowed, but np.nan is not. 

        """

        if bounds is None:
            bounds = np.array((-np.inf,np.inf))

        try:
            bounds = np.array(bounds,dtype=float)
        except Exception as e:
            raise ValueError(err_msg) from e
        
        if len(bounds.shape) == 0 or bounds.shape[0] != 2:
            raise ValueError(err_msg)
        
        num_nan = np.sum(np.isnan(bounds))
        if num_nan > 0:
            raise ValueError(err_msg)
            
        # Set any bounds very close to zero to zero.
        equiv_to_zero = np.finfo(bounds.dtype).resolution
        bounds[np.abs(bounds) < equiv_to_zero] = 0.0

        # Make sure upper bound is above the lower bound
        if bounds[1] <= bounds[0]:
            raise ValueError(err_msg)

        # Shift existing guess if necessary
        if self.guess is not None:
        
            if self.guess < bounds[0]:
                new_guess = bounds[0]
            elif self.guess > bounds[1]:
                new_guess = bounds[1]
            else:
                new_guess = None

            if new_guess is not None:

                w = f"The previous guess ({self.guess}) is outside the new\n"
                w += f"bounds ({bounds}). The guess has been updated to\n"
                w += f"'{self.guess}'.\n"
                warnings.warn(w,UserWarning)

                self.guess = new_guess

        self._bounds = bounds
        self._clear_fit_result()

    #--------------------------------------------------------------------------
    # priors for fit.

    @property
    def prior(self):
        """
        Parameter prior (mean, std).  Either numpy array of two elements or None.
        """

        try:
            return self._prior
        except AttributeError:
            return None
        
    @prior.setter
    def prior(self,prior):
        """
        Set the prior.
        """

        err_msg = \
        """

        prior should be list-like, with two floats. The first entry is the 
        mean of a gaussian prior; the second is the standard deviation.
        
        + If prior == None, the prior is set to [np.nan,np.nan], which will 
          cause a Bayesian inference to use uniform priors
        + The standard deviation must be positive.
        + np.nan values are allowed, but np.inf is not. 

        """

        if prior is None:
            prior = np.nan*np.ones(2,dtype=float)

        # Can be coerced to float array
        try:
            prior = np.array(prior,dtype=float)
        except Exception as e:
            raise ValueError(err_msg) from e
        
        # two elements
        if len(prior.shape) == 0 or prior.shape[0] != 2:
            raise ValueError(err_msg)

        # no infinities allowed
        num_inf = np.sum(np.isinf(prior))
        if num_inf > 0:
            raise ValueError(err_msg)

        # stdev must be >= 0
        if prior[1] <= 0:
            raise ValueError(err_msg)
        
        self._prior = prior
        self._clear_fit_result()


    #--------------------------------------------------------------------------
    # Properties that are set by the fitter, but not the user.

    @property
    def value(self):
        """
        Value of the parameter.
        """

        try:
            return self._value
        except AttributeError:
            return None


    @property
    def stdev(self):
        """
        Standard deviation on the parameter.
        """

        try:
            return self._stdev
        except AttributeError:
            return None

    @property
    def ninetyfive(self):
        """
        95% confidence interval on the parameter.
        """

        try:
            return self._ninetyfive
        except AttributeError:
            return None

    @property
    def is_fit_result(self):
        """
        Whether the value shown is the result of a fit.
        """

        return self._is_fit_result


    def load_fit_result(self,fitter,param_number):
        """
        Update the standard deviation, ninetyfive, parameter value, from a
        successful fit.

        Parameters
        ----------
        fitter : dataprob.Fitter
            fit with results
        param_number : int
            number corresponding to this parameter in the fit parameter vector.
        """

        if fitter.success:
            self._value = fitter.estimate[param_number]
            self._stdev = fitter.stdev[param_number]
            if fitter.ninetyfive is not None:
                self._ninetyfive = np.array([fitter.ninetyfive[0,param_number],
                                             fitter.ninetyfive[1,param_number]])
            else:
                self._ninetyfive = np.array([np.nan,np.nan])
            self._is_fit_result = True

    def _clear_fit_result(self):
        """
        Clear the fit result.  Called when attributes controlling the fit
        are modified to avoid inconsistency.
        """

        self._value = self.guess
        self._stdev = None
        self._ninetyfive = None
        self._is_fit_result = False
