__description__ = \
"""
Main class for holding fit parameters, including guesses, values, ranges, etc.
"""
__date__ = "2016-09-02"
__author__ = "Michael J. Harms"

import copy
import numpy as np

import warnings

_INFINITY_PROXY = 1e9
_CLOSE_TO_ZERO = 1e6

class FitParameter:
    """
    Class for storing and manipulating fit parameters.

    Fit control attributes:
    -----------------------

    These attributes can be set on __init__ or directly via instance.xxx = value.
    If any of these values are set, the fit result (stored in .value) is wiped
    out to avoid inconsistency between the fit control parameters and the
    resulting fit.

        + .guess: (float) guess value for fit
        + .fixed: (bool) whether to float parameter or not
        + .bounds: [float,float] lower and upper bounds for parameter

    Fit result attributes:
    ----------------------

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

    Metadata attributes:
    --------------------

    This attribute can be set on __init__ or directly via instance.name = value.
    The value is used for convenience only and is ignored by other methods.

        + .name: (str) name of fit parameter
    """

    def __init__(self,name,guess=None,fixed=False,bounds=None):
        """
        Initialize class.  Parameters:

        name: name of parameter (string)
        guess: parameter guess (float).
               If None:
                    + If bounds are given:
                        - If lower and upper bounds have the same sign, the guess
                          will be placed at the geometric mean of the two bounds.
                        - Otherwise, the guess is placed at the arithmetic mean
                          of the two bounds.
                    + If bounds are not given, guess will be set to 1.0.
        fixed: whether or not the parameter is fixed (bool)
        bounds: bounds on fit for parameter (list-like object of 2 floats). If
                None, bounds will be set to (None,None).  If (None,5), no lower
                bound, upper bound of 5.
        """

        # Setting must be in this order. If no guess is specified, the guess
        # is made based on the bounds.
        self._is_fit_result = False
        self.bounds = bounds
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
        Set the guess.  If None and no bounds are specified, set to 1.0.  If
        None and bounds are specified, set to geometric mean of the bounds.
        If one of the bounds is infinte in this latter case, set that bound to
        {_INFINITY_PROXY} for the purpose of the midpoint calculation.
        """

        if guess is not None:

            try:
                guess = float(guess)
            except (ValueError,TypeError):
                err = f"parameter guess '{guess}' cannot be interpretable as a float\n"
                raise ValueError(err)

        else:

            if self.bounds[0] == -np.inf and self.bounds[1] == np.inf:
                guess = 1.0
            else:

                # If we have non-infinite bounds, take the geometric midpoint of
                # the bounds as the guess
                if self.bounds[0] > -np.inf:
                    lower_bound = self.bounds[0]
                    lower_sign = np.abs(lower_bound)/lower_bound
                else:
                    lower_bound = -_INFINITY_PROXY
                    lower_sign = -1

                if self.bounds[1] < np.inf:
                    upper_bound = self.bounds[1]
                    upper_sign = np.abs(upper_bound)/upper_bound
                else:
                    upper_bound = _INFINITY_PROXY
                    upper_sign = 1

                # If the bounds have the same sign, use geometric mean
                if upper_sign == lower_sign:
                    l = np.log(np.abs(lower_bound))
                    u = np.log(np.abs(upper_bound))
                    guess = upper_sign * np.exp((u + l)/2)


                # If they have opposite signs, use the arithmetic mean
                elif upper_sign > lower_sign:
                    guess = (upper_bound + lower_bound)/2

                # This should *never* be true. This would imply the upper
                # bound is negative and the lower bound is positive.
                else:
                    err = f"Could not set guess automatically: lower bound \n"
                    err += f"({self.bounds[0]}) somehow above upper bound ({self.bound[1]}).\n"
                    err += "this is probably a bug in the likelihood code. You\n"
                    err += "can file a bug report at https://github.com/harmslab/likelihood\n"
                    raise RuntimeError(err)

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

        self._fixed = bool(bool_value)

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

        if bounds is not None:

            try:
                if len(bounds) != 2:
                    raise TypeError

                bounds = np.array(bounds,dtype=float)

            except TypeError:
                err = "Bounds must be list-like object of length 2\n"
                raise ValueError(err)

        else:
            bounds = np.array((-np.inf,np.inf))

        # Set bounds very close to zero to zero.
        equiv_to_zero = np.finfo(bounds.dtype).tiny*_CLOSE_TO_ZERO
        bounds[np.abs(bounds) < equiv_to_zero] = 0.0

        # Make sure upper bound is above the lower bound
        if bounds[1] <= bounds[0]:
            err = f"upper bound ({bounds[1]}) must be greater than lower bound ({bounds[0]})"
            raise ValueError(err)

        if self.guess is not None:
            if self.guess < bounds[0]:
                old_guess = self.guess
                self.guess = bounds[0]
            elif self.guess > bounds[1]:
                old_guess = self.guess
                self.guess = bounds[1]
            else:
                old_guess = None

            if old_guess is not None:
                w = f"The previous guess ({old_guess}) is outside the new bounds ({bounds})\n"
                w += f"Guess has been updated to {self.guess}\n"
                warnings.warn(w,UserWarning)

        self._bounds = bounds
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
        Update standard deviation, ninetyfive, parameter value, from a successful
        fit.

        fitter: Fitter instance.
        param_number: number corresponding to this parameter in the fit parameter
                      vector.
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
        Cleared the fit result.  Called when attributes controlling the fit
        are modified to avoid inconsistency.
        """

        self._value = self.guess
        self._stdev = None
        self._ninetyfive = None
        self._is_fit_result = False
