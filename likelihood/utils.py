
class ModelWrapper:
    """
    Wrap a model for use in the likelihood api.  The model is expected
    to be a python function that:

    1. Calculates whatever observable you are using (i.e. the "MODEL" bit in the
       L = P(DATA|MODEL) ).
    2. Takes the fit parameters as its first n arguments.
    3. T
    """

    def __init__(self,real_model,args=[],kwargs={}):

        self._real_model = real_model
        self._model_args = args
        self._model_kwargs = kwargs

    def observable(self,params):
        """
        """

        return self._real_model(*params,
                                *self._model_args,
                                **self._model_kwargs)
