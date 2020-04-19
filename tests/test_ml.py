
import likelihood
import numpy as np
import os


class TestMLFitter:

    def test_init(self,binding_curve_test_data):

        self._fitter_instance = likelihood.MLFitter()
        assert self._fitter_instance.fit_type == "maximum likelihood"

    def test_fit(self,binding_curve_test_data):

        FIT_TOLERANCE = 0.1

        input_params = np.array(binding_curve_test_data[0])
        binding_curve_model = binding_curve_test_data[1]
        test_data_frames = binding_curve_test_data[2]

        f = likelihood.MLFitter()

        for df in test_data_frames:

            lm = likelihood.ModelWrapper(binding_curve_model,
                                         **{"X":df.X})
            f.fit(lm.observable,input_params,df.Y)

            assert f.success
            assert np.allclose(f.estimate,
                               input_params,
                               rtol=FIT_TOLERANCE,
                               atol=FIT_TOLERANCE*input_params)


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
