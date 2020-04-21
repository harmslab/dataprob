
from .fit_param import FitParameter

import numpy as np

import inspect

class ModelWrapper:
    """
    Wraps a model.
    """

    #_mw_ pre-prepended to internal variables to avoid contaminating
    # class namespace when we start sticking arbitrary model arguments
    # into self.__dict__

    # Has to be defined across class because we are going to hijack
    # __setattr__ and need to look inside this as soon as we start
    # setting attributes.
    _mw_fit_parameters = {}
    _mw_other_arguments = {}

    def __init__(self,model_to_fit):
        """
        """

        self._model_to_fit = model_to_fit

        self._mw_load_model()

    def _mw_load_model(self):
        """
        """

        self._mw_fit_parameters = {}
        self._mw_other_arguments = {}

        getting_fit_params = True
        self._mw_signature = inspect.signature(self._model_to_fit)
        for p in self._mw_signature.parameters:

            # Make sure that this parameter name isn't already being used by
            # the wrapper.
            try:
                self.__dict__[p]
                raise ValueError
            except ValueError:
                err = f"Parameter name '{p}' reserved by this class.\n"
                err += "Please change the argument name in your function.\n"
                raise ValueError(err)
            except KeyError:
                pass

            # If we hit args or kwargs, stop looking for fittable parameters
            if p in ['args','kwargs']:
                getting_fit_params = False

            # Try to turn the argument default into a guess.  If this fails b/c
            # default is not coercable into a float, stop trying to grab fit
            # parameters.
            if getting_fit_params or get_specific_params:

                if self._mw_signature.parameters[p].default == inspect._empty:
                    guess = None
                else:
                    guess = self._mw_signature.parameters[p].default
                    try:
                        guess = np.float(guess)
                    except (TypeError,ValueError):
                        getting_fit_params = False

            # If we are still getting fit parameters, record this as a fit parameter
            if getting_fit_params:
                self._mw_fit_parameters[p] = FitParameter(name=p,guess=guess)
                self.__dict__[p] = self._mw_fit_parameters[p]

            # Otherwise, this is a standard model argument
            else:
                self._mw_other_arguments[p] = self._mw_signature.parameters[p].default
                self.__dict__[p] = self._mw_other_arguments[p]


    def __setattr__(self, key, value):
        """
        Hijack __setattr__ so setting the value for fit parameters
        updates the fit guess.
        """


        # We're setting the guess of the fit parameter
        if key in self._mw_fit_parameters.keys():

            self._mw_fit_parameters[key].guess = value

        # We're setting another argument
        elif key in self._mw_other_arguments.keys():

            self._mw_other_arguments[key] = value

        # Otherwise, just set it like normal
        else:
            super(ModelWrapper, self).__setattr__(key, value)
            return


    def _update_parameter_map(self):
        """
        Update the map between the parameter vector that will be passed in to
        the fitter and the parameters in this wrapper.
        """

        self._param_to_p_map = []
        self._mw_kwargs = {}
        counter = 0
        for p in self._mw_fit_parameters.keys():
            if self._mw_fit_parameters[p].fixed:
                self._mw_kwargs[p] = self.fit_parameters[p].value
            else:
                self._mw_kwargs[p] = None
                self._param_to_p_map.append(p)
                counter += 1

        self._mw_kwargs.update(self._mw_other_arguments)


    def _mw_observable(self,params):
        """
        Actual function called by the fitter.
        """

        for i in range(len(params)):
            self._mw_kwargs[self._param_to_p_map[i]] = params[i]

        return self._model_to_fit(**self._mw_kwargs)


    @property
    def model(self):

        self._update_parameter_map()

        return self._mw_observable

    @property
    def guesses(self):

        self._update_parameter_map()

        guesses = []
        for p in self._param_to_p_map:
            guesses.append(self.fit_parameters[p].guess)

        return np.array(guesses)

    @property
    def bounds(self):

        self._update_parameter_map()

        bounds = [[],[]]
        for p in self._param_to_p_map:
            bounds[0].append(self.fit_parameters[p].bounds[0])
            bounds[1].append(self.fit_parameters[p].bounds[1])

        return np.array(bounds)

    @property
    def param_names(self):

        self._update_parameter_map()

        param_names = []
        for p in self._param_to_p_map:
            param_names.append(self.fit_parameters[p].name)

        return param_names[:]


    @property
    def fit_parameters(self):
        """
        """

        return self._mw_fit_parameters

    @property
    def other_arguments(self):
        """
        """

        return self._mw_other_arguments
