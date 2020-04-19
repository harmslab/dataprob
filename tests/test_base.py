
import likelihood
import numpy as np

class TestBaseFitter:

    def test_init(self,binding_curve_test_data):

        self._f = likelihood.base.Fitter()
        assert self._f.fit_type == ""




    def test_fit(self,binding_curve_test_data,fit_tolerance_fixture):
        """
        Test the ability to fit the test data in binding_curve_test_data.
        """

        # Construct fitter instance
        self.test_init(binding_curve_test_data)

        # For base class, return and don't actually do a fit.
        if self._f.fit_type == "":
            return

        input_params = np.array(binding_curve_test_data[0])
        binding_curve_model = binding_curve_test_data[1]
        test_df = binding_curve_test_data[2]

        # Do the fit on this data frame
        lm = likelihood.ModelWrapper(binding_curve_model,
                                     **{"X":test_df.X})
        self._f.fit(lm.observable,input_params,test_df.Y)

        # Fit should converge and give right value
        assert self._f.success
        assert np.allclose(self._f.estimate,
                           input_params,
                           rtol=fit_tolerance_fixture,
                           atol=fit_tolerance_fixture*input_params)

    def test_unweighted_residuals(self,binding_curve_test_data):

        # Construct fitter instance
        self.test_init(binding_curve_test_data)
