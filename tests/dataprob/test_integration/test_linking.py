
import pytest
import dataprob
import numpy as np

def _covarying_model(m1=1,m2=1,x=None):

    return m1*m2*x

def _covarying_model_vec(params,x=None):

    return params[0]*params[1]*x


def test_linking_normal():

    x = np.arange(0,10)
    y_obs = x*4 
    y_std = 0.000001*np.ones(10)

    # Fit model with two floating parameters
    f = dataprob.setup(some_function=_covarying_model,
                       method="ml",
                       non_fit_kwargs={"x":x})

    # Should throw warning because parameters perfectly co-vary
    with pytest.warns():
        f.fit(y_obs=y_obs,
            y_std=y_std)
    
    # Now fix m2
    f = dataprob.setup(some_function=_covarying_model,
                       method="ml",
                       non_fit_kwargs={"x":x})
    f.param_df.loc["m2","fixed"] = True
    f.param_df.loc["m2","guess"] = 1

    f.fit(y_obs=y_obs,
          y_std=y_std)
    
    assert np.isclose(f.fit_df.loc["m1","estimate"],4.0)

    # Now link m1 to m2
    f = dataprob.setup(some_function=_covarying_model,
                       method="ml",
                       non_fit_kwargs={"x":x})
    f.param_df.loc["m2","parent"] = "m1"

    f.fit(y_obs=y_obs,
          y_std=y_std)
    
    assert np.isclose(f.fit_df.loc["m1","estimate"],2.0)
    assert np.isclose(f.fit_df.loc["m2","estimate"],2.0)
    
    m1_values = np.array(f.fit_df.loc["m1",["estimate","std","low_95","high_95"]],dtype=float)
    m2_values = np.array(f.fit_df.loc["m2",["estimate","std","low_95","high_95"]],dtype=float)
    assert np.array_equal(m1_values,m2_values)
    
    
def test_linking_vector():

    x = np.arange(0,10)
    y_obs = x*4 
    y_std = 0.000001*np.ones(10)

    # Fit model with two floating parameters
    f = dataprob.setup(some_function=_covarying_model_vec,
                       method="ml",
                       fit_parameters=["m1","m2"],
                       vector_first_arg=True,
                       non_fit_kwargs={"x":x})

    # Should throw warning because parameters perfectly co-vary
    with pytest.warns():
        f.fit(y_obs=y_obs,
            y_std=y_std)
    
    # Now fix m2
    f = dataprob.setup(some_function=_covarying_model_vec,
                       method="ml",
                       fit_parameters=["m1","m2"],
                       vector_first_arg=True,
                       non_fit_kwargs={"x":x})
    f.param_df.loc["m2","fixed"] = True
    f.param_df.loc["m2","guess"] = 1

    f.fit(y_obs=y_obs,
          y_std=y_std)
    
    assert np.isclose(f.fit_df.loc["m1","estimate"],4.0)

    # Now link m1 to m2
    f = dataprob.setup(some_function=_covarying_model_vec,
                       method="ml",
                       fit_parameters=["m1","m2"],
                       vector_first_arg=True,
                       non_fit_kwargs={"x":x})
    f.param_df.loc[:,"guess"] = [1,4]
    f.param_df.loc["m2","parent"] = "m1"
    
    f.fit(y_obs=y_obs,
          y_std=y_std)
    
    assert np.isclose(f.fit_df.loc["m1","estimate"],2.0)
    assert np.isclose(f.fit_df.loc["m2","estimate"],2.0)
    
    m1_values = np.array(f.fit_df.loc["m1",["estimate","std","low_95","high_95"]],dtype=float)
    m2_values = np.array(f.fit_df.loc["m2",["estimate","std","low_95","high_95"]],dtype=float)
    assert np.array_equal(m1_values,m2_values)
    



    