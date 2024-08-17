__description__ = \
"""
Key public functions and methods for dataprob library.
"""

from .fitters.ml import MLFitter
from .fitters.bootstrap import BootstrapFitter
from .fitters.bayesian.bayesian_sampler import BayesianSampler

from . import plot

from .__version__ import __version__