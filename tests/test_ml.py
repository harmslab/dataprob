
import likelihood
import numpy as np
import os

from test_base import TestBaseFitter

class TestMLFitter(TestBaseFitter):

    def test_init(self,binding_curve_test_data):

        self._f = likelihood.MLFitter()
        assert self._f.fit_type == "maximum likelihood"


    def test_unweighted_residuals(self):
        pass

    def test_weighted_residuals(self):
        pass

    def test_ln_like(self):
        pass

    def test_estimate(self):
        """
        Estimates of fit parameters.
        """

        pass

    def test_stdev(self):
        """
        Standard deviations on estimates of fit parameters.
        """

        pass

    def test_ninetyfive(self):
        """
        Ninety-five perecent confidence intervals on the estimates.
        """

        pass

    def test_fit_result(self):
        """
        Full fit results (will depend on exact fit type what is placed here).
        """
        pass


    def test_success(self):
        """
        Whether the fit was successful.
        """

        pass

    def test_fit_info(self):
        """
        Information about fit run.  Should be redfined in subclass.
        """

        pass

    def test_corner_plot(self):
        pass

    def test_samples(self):
        """
        Samples from stochastic fits.
        """

        pass
