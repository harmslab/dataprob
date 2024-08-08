__description__ = \
"""
Key public functions and methods for dataprob library.
"""

from .model_wrapper.wrap_function import wrap_function

from .fitters.ml import MLFitter
from .fitters.bootstrap import BootstrapFitter
from .fitters.bayesian import BayesianFitter


