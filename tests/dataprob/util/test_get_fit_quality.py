import pytest

from dataprob.util.get_fit_quality import _get_num_obs
from dataprob.util.get_fit_quality import _get_num_param
from dataprob.util.get_fit_quality import _get_lnL
from dataprob.util.get_fit_quality import _get_chi2
from dataprob.util.get_fit_quality import _get_reduced_chi2
from dataprob.util.get_fit_quality import _get_mean0_ttest
from dataprob.util.get_fit_quality import _get_durbin_watson
from dataprob.util.get_fit_quality import _get_ljung_box
from dataprob.util.get_fit_quality import get_fit_quality

import numpy as np
import pandas as pd

def test__get_num_obs():

    out_dict = {"name":[],
                "description":[],
                "is_good":[],
                "value":[],
                "message":[]}
    
    out_dict = _get_num_obs(5,out_dict)
    assert out_dict["name"][0] == "num_obs"
    assert out_dict["description"][0] == "number of observations"
    assert out_dict["value"][0] == 5
    assert out_dict["message"][0] == ""
    assert out_dict["is_good"][0] is True

def test__get_num_param():

    out_dict = {"name":[],
                "description":[],
                "is_good":[],
                "value":[],
                "message":[]}
    
    # XX uncomment lines below after pip install

    out_dict = _get_num_param(num_param=1,num_obs=5,out_dict=out_dict)
    assert out_dict["name"][0] == "num_param"
    assert out_dict["description"][0] == "number of fit parameters"
    assert out_dict["value"][0] == 1
    assert out_dict["message"][0] == "There are 4 more observations than fit parameters."
    assert out_dict["is_good"][0] is True

    out_dict = _get_num_param(num_param=6,num_obs=5,out_dict=out_dict)
    assert out_dict["name"][1] == "num_param"
    assert out_dict["description"][0] == "number of fit parameters"
    assert out_dict["value"][1] == 6
    assert out_dict["message"][1] == f"There are not enough observations (5) to support the number of fit parameters (6)."
    assert out_dict["is_good"][1] is False

def test__get_lnL():
    
    out_dict = {"name":[],
                "description":[],
                "is_good":[],
                "value":[],
                "message":[]}

    out_dict = _get_lnL(-5,out_dict)
    assert out_dict["name"][0] == "lnL"
    assert out_dict["description"][0] == "log likelihood"
    assert out_dict["value"][0] == -5
    assert out_dict["message"][0] == ""
    assert out_dict["is_good"][0] is True

def test__get_chi2():

    out_dict = {"name":[],
                "description":[],
                "is_good":[],
                "value":[],
                "message":[]}
    
    # residuals are not random --> chi2 should fail
    r = np.arange(100)
    num_param = 2
    out_dict = _get_chi2(residuals=r,
                         num_param=num_param,
                         out_dict=out_dict)
    assert out_dict["name"][0] == "chi2"
    assert out_dict["description"][0] == "chi^2 goodness-of-fit"
    assert out_dict["value"][0] < 0.05
    assert out_dict["message"][0] != "" #<- don't really check this because it's a pain
    assert out_dict["is_good"][0] is False
    
    # residuals are random --> chi2 should succeed
    r = np.array([-2,-1,-1,-1,-1,0,0,0,0,0,0,0,0,0,0,1,1,1,1,2])
    num_param = 2
    out_dict = _get_chi2(residuals=r,
                         num_param=num_param,
                         out_dict=out_dict)
    assert out_dict["name"][1] == "chi2"
    assert out_dict["description"][1] == "chi^2 goodness-of-fit"
    assert out_dict["value"][1] > 0.05
    assert out_dict["message"][1] != "" #<- don't really check this because it's a pain
    assert out_dict["is_good"][1] is True
    
def test__get_reduced_chi2():
    
    out_dict = {"name":[],
                "description":[],
                "is_good":[],
                "value":[],
                "message":[]}

    r = 0.1*np.ones(10)
    n = 1
    out_dict = _get_reduced_chi2(residuals=r,
                                 num_param=n,
                                 out_dict=out_dict)
    
    assert out_dict["name"][0] == "reduced_chi2"
    assert out_dict["description"][0] == "reduced chi^2"
    assert np.isclose(out_dict["value"][0],0.0125)
    assert out_dict["message"][0] == "A reduced chi^2 value of 0.013 may mean the model is overfit or the that the uncertainty on each point is overestimated."
    assert out_dict["is_good"][0] is False

    r = np.ones(4)
    n = 1
    out_dict = _get_reduced_chi2(residuals=r,
                                 num_param=n,
                                 out_dict=out_dict)
    
    assert out_dict["name"][1] == "reduced_chi2"
    assert out_dict["description"][1] == "reduced chi^2"
    assert out_dict["value"][1] == 2.0
    assert out_dict["message"][1] == "A reduced chi^2 value of 2.000 may mean the model does not describe the data well or that the uncertainty on each point is underestimated."
    assert out_dict["is_good"][1] is False
    
    r = 0.89*np.ones(10)
    n = 1
    out_dict = _get_reduced_chi2(residuals=r,
                                 num_param=n,
                                 out_dict=out_dict)
    
    assert out_dict["name"][2] == "reduced_chi2"
    assert out_dict["description"][2] == "reduced chi^2"
    assert np.isclose(np.round(out_dict["value"][2],3),0.990)
    assert out_dict["message"][2] == "A reduced chi^2 value of 0.990 is consistent with the model describing the data well."
    assert out_dict["is_good"][2] is True
    
def test__get_mean0_ttest():

    out_dict = {"name":[],
                "description":[],
                "is_good":[],
                "value":[],
                "message":[]}
    
    r = np.array([1,0,-1,1,0,-1])
    out_dict = _get_mean0_ttest(residuals=r,out_dict=out_dict)
    assert out_dict["name"][0] == "mean0_resid"
    assert out_dict["description"][0] == "t-test for residual mean != 0"
    assert out_dict["value"][0] > 0.05
    assert out_dict["message"][0] != ""
    assert out_dict["is_good"][0] is True
    
    r = np.array([1,1.1,1,1.1,0.99,1.1])
    out_dict = _get_mean0_ttest(residuals=r,out_dict=out_dict)
    assert out_dict["name"][1] == "mean0_resid"
    assert out_dict["description"][1] == "t-test for residual mean != 0"
    assert out_dict["value"][1] < 0.05
    assert out_dict["message"][1] != ""
    assert out_dict["is_good"][1] is False

def test__get_durbin_watson():

    out_dict = {"name":[],
                "description":[],
                "is_good":[],
                "value":[],
                "message":[]}
    
    r = np.random.normal(0,1,size=10000)
    out_dict = _get_durbin_watson(residuals=r,out_dict=out_dict)

    assert out_dict["name"][0] == "durbin-watson"
    assert out_dict["description"][0] == "Durbin-Watson test for correlated residuals"
    assert out_dict["value"][0] > 1
    assert out_dict["value"][0] < 3
    assert out_dict["message"][0] != ""
    assert out_dict["is_good"][0] is True

    r = np.array([1,-1,1,-1,1,-1,1,-1,1,-1,1,-1])
    out_dict = _get_durbin_watson(residuals=r,out_dict=out_dict)

    assert out_dict["name"][1] == "durbin-watson"
    assert out_dict["description"][1] == "Durbin-Watson test for correlated residuals"
    assert out_dict["value"][1] > 3
    assert out_dict["message"][1] != ""
    assert out_dict["is_good"][1] is False

    r = np.arange(100)
    out_dict = _get_durbin_watson(residuals=r,out_dict=out_dict)

    assert out_dict["name"][2] == "durbin-watson"
    assert out_dict["description"][2] == "Durbin-Watson test for correlated residuals"
    assert out_dict["value"][2] < 1
    assert out_dict["message"][2] != ""
    assert out_dict["is_good"][2] is False

def test__get_ljung_box():

    out_dict = {"name":[],
                "description":[],
                "is_good":[],
                "value":[],
                "message":[]}
    
    r = np.arange(100)
    n = 1
    out_dict = _get_ljung_box(residuals=r,
                              num_param=n,
                              out_dict=out_dict)

    assert out_dict["name"][0] == "ljung-box"
    assert out_dict["description"][0] == "Ljung-Box test for correlated residuals"
    assert out_dict["value"][0] < 0.05
    assert out_dict["message"][0] != ""
    assert out_dict["is_good"][0] is False


    r = np.array([-0.05495277, -0.59843925,  0.80701057,  0.51253132, 0.8113159 ,
                  -1.25294175,  1.63693971, -0.88914536, -1.06075899, -0.84673795])
    n = 1
    out_dict = _get_ljung_box(residuals=r,
                              num_param=n,
                              out_dict=out_dict)

    assert out_dict["name"][1] == "ljung-box"
    assert out_dict["description"][1] == "Ljung-Box test for correlated residuals"
    assert out_dict["value"][1] > 0.05
    assert out_dict["message"][1] != ""
    assert out_dict["is_good"][1] is True

def test_get_fit_quality():


    r = np.random.normal(0,1,size=10000)
    n = 1
    lnL = -5

    # Just make sure this runs; output checked in other tests
    out = get_fit_quality(residuals=r,
                          num_param=n,
                          lnL=lnL)
    assert issubclass(type(out),pd.DataFrame)
    assert len(out) == 8
    assert np.array_equal(out.columns,["description","is_good","value","message"])
    assert out.loc["lnL","value"] == -5
    assert out.loc["num_param","value"] == 1

    
