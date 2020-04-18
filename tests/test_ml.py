
import likelihood
import os

def test_init(binding_curves):

    f = likelihood.MLFitter()

    assert f.fit_type == "maximum likelihood"
