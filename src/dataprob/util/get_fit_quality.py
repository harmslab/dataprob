
from dataprob.util.stats import chi2
from dataprob.util.stats import chi2_reduced
from dataprob.util.stats import durbin_watson
from dataprob.util.stats import ljung_box

import pandas as pd
from scipy import stats

def _get_success(success,out_dict):
    """
    Get whether the fit was successful and append to out_dict. 
    
    Parameters
    ----------
    success : bool
        whether or not the fit was successful, as determined by the Fitter
        object. 
    out_dict : dict
        dictionary with 'name', 'description', 'value', 'is_good', and 'msg' 
        keys pointing to list values

    Returns
    out_dict : dict
        out_dict updated with results of this quality test.
    """

    out_dict["name"].append("success")
    out_dict["description"].append("fit success status")
    out_dict["value"].append(success)
    out_dict["message"].append("")
    out_dict["is_good"].append(success)
    
    return out_dict

def _get_num_obs(num_obs,out_dict):
    """
    Get the number of observations and append to out_dict. 
    
    Parameters
    ----------
    num_obs : int
        number of observations
    out_dict : dict
        dictionary with 'name', 'description', 'value', 'is_good', and 'msg' 
        keys pointing to list values

    Returns
    out_dict : dict
        out_dict updated with results of this quality test.
    """

    out_dict["name"].append("num_obs")
    out_dict["description"].append("number of observations")
    out_dict["value"].append(num_obs)
    out_dict["message"].append("")
    out_dict["is_good"].append(True)

    return out_dict

def _get_num_param(num_param,num_obs,out_dict):
    """
    Get the number of fit parameters, interpret it, and append the results
    to out_dict. 
    
    Parameters
    ----------
    num_param : int
        number of fit parameters
    num_obs : int
        number of observations
    out_dict : dict
        dictionary with 'name', 'description', 'value', 'is_good', and 'msg' 
        keys pointing to list values

    Returns
    out_dict : dict
        out_dict updated with results of this quality test.
    """

    out_dict["name"].append("num_param")
    out_dict["description"].append("number of fit parameters")
    out_dict["value"].append(num_param)
    
    is_good = num_param < num_obs
    out_dict["is_good"].append(is_good)
    
    if is_good:
        msg = f"There are {num_obs - num_param} more observations than "
        msg += "fit parameters."
    else:
        msg = f"There are not enough observations ({num_obs}) to support "
        msg += f"the number of fit parameters ({num_param})."

    out_dict["message"].append(msg)

    return out_dict
    
def _get_lnL(lnL,out_dict):
    """
    Append lnL to out_dict. 
    
    Parameters
    ----------
    lnL : float
        log likelihood at fit parameter estimates
    out_dict : dict
        dictionary with 'name', 'description', 'value', 'is_good', and 'msg' 
        keys pointing to list values

    Returns
    out_dict : dict
        out_dict updated with results of this quality test.
    """
    out_dict["name"].append("lnL")
    out_dict["description"].append("log likelihood")
    out_dict["value"].append(lnL)
    out_dict["is_good"].append(True)
    out_dict["message"].append("")

    return out_dict

def _get_chi2(residuals,num_param,out_dict):
    """
    Get the chi^2 goodness-of-fit p-value, interpret it, and append the results
    to out_dict. 
    
    Parameters
    ----------
    residuals : numpy.ndarray
        float array holding weighted fit residuals
    num_param : int
        number of fit parameters
    out_dict : dict
        dictionary with 'name', 'description', 'value', 'is_good', and 'msg' 
        keys pointing to list values

    Returns
    out_dict : dict
        out_dict updated with results of this quality test.
    """

    name = "chi2"
    txt = "chi^2 goodness-of-fit"
    value = chi2(residuals,num_param=num_param)
    if value < 0.05:
        msg = f"A p-value of {value:.3e} for the a goodness-of-fit Chi^2 test "
        msg += "is evidence that the model does not describe the model well "
        msg += "(model violation). Look for regions where your model "
        msg += "systematically deviates from your data."
        is_good = False
    else:
        msg = f"A p-value of {value:.3e} for the a goodness-of-fit Chi^2 test "
        msg += "is consistent with the model describing the data well."
        is_good = True
    
    out_dict["name"].append(name)
    out_dict["description"].append(txt)
    out_dict["value"].append(value)
    out_dict["is_good"].append(is_good)
    out_dict["message"].append(msg)

    return out_dict

def _get_reduced_chi2(residuals,num_param,out_dict):
    """
    Get the reduced chi^2 value, interpret it, and append the results to out_dict. 
    
    Parameters
    ----------
    residuals : numpy.ndarray
        float array holding weighted fit residuals
    num_param : int
        number of fit parameters
    out_dict : dict
        dictionary with 'name', 'description', 'value', 'is_good', and 'msg' 
        keys pointing to list values

    Returns
    out_dict : dict
        out_dict updated with results of this quality test.
    """

    name = "reduced_chi2"
    txt = "reduced chi^2"
    value = chi2_reduced(residuals,num_param=num_param)
    if value < 0.75:
        msg = f"A reduced chi^2 value of {value:.3f} may mean the model is "
        msg += "overfit or the that the uncertainty on each point is "
        msg += "overestimated."
        is_good = False
    elif value > 1.25:
        msg = f"A reduced chi^2 value of {value:.3f} may mean the model "
        msg += "does not describe the data well or that the uncertainty on "
        msg += "each point is underestimated."
        is_good = False
    else:
        msg = f"A reduced chi^2 value of {value:.3f} is consistent with "
        msg += "the model describing the data well."
        is_good = True

    out_dict["name"].append(name)
    out_dict["description"].append(txt)
    out_dict["value"].append(value)
    out_dict["is_good"].append(is_good)
    out_dict["message"].append(msg)

    return out_dict

def _get_mean0_ttest(residuals,out_dict):
    """
    Run test for residual mean centered at zero, interpret it, and append the
    results to out_dict. 
    
    Parameters
    ----------
    residuals : numpy.ndarray
        float array holding weighted fit residuals
    out_dict : dict
        dictionary with 'name', 'description', 'value', 'is_good', and 'msg' 
        keys pointing to list values

    Returns
    out_dict : dict
        out_dict updated with results of this quality test.
    """

    name = "mean0_resid"
    txt = "t-test for residual mean != 0"
    value = stats.ttest_1samp(residuals,popmean=0).pvalue
    if value < 0.05:
        msg = f"A p-value of {value:.3e} for the one-sample t-test is "
        msg += "evidence for fit residuals with a non-zero mean. "
        msg += "This may mean your model does not describe your data well "
        msg += "(model violation). Look for regions where your model "
        msg += "systematically deviates from your data."
        is_good = False
    else:
        msg = f"A p-value of {value:.3e} for the one-sample t-test is "
        msg += "consistent with the residuals having a mean of 0."
        is_good = True

    out_dict["name"].append(name)
    out_dict["description"].append(txt)
    out_dict["value"].append(value)
    out_dict["is_good"].append(is_good)
    out_dict["message"].append(msg)

    return out_dict


def _get_durbin_watson(residuals,out_dict):
    """
    Run Durbin-Watson test, interpret it, and append the results to out_dict. 
    
    Parameters
    ----------
    residuals : numpy.ndarray
        float array holding weighted fit residuals
    out_dict : dict
        dictionary with 'name', 'description', 'value', 'is_good', and 'msg' 
        keys pointing to list values

    Returns
    out_dict : dict
        out_dict updated with results of this quality test.
    """

    name = "durbin-watson"
    txt = "Durbin-Watson test for correlated residuals"      
    value, _ = durbin_watson(residuals)

    if value <= 1:
        msg = f"A Durbin-Watson test-statistic of {value:.3f} is "
        msg += "is evidence for positively correlated fit residuals. "
        msg += "This may mean your model does not describe your data well "
        msg += "(model violation). Look for regions where your model "
        msg += "systematically deviates from your data."
        is_good = False
    elif value >= 3:
        msg = f"A Durbin-Watson test-statistic of {value:.3f} is "
        msg += "evidence for negatively correlated fit residuals. "
        msg += "This may mean your model does not describe your data well "
        msg += "(model violation). Look for regions where your model "
        msg += "systematically deviates from your data."
        is_good = False
    else:
        msg = f"A Durbin-Watson test-statistic of {value:.3f} is "
        msg += "consistent with un-correlated fit residuals."
        is_good = True

    out_dict["name"].append(name)
    out_dict["description"].append(txt)
    out_dict["value"].append(value)
    out_dict["is_good"].append(is_good)
    out_dict["message"].append(msg)

    return out_dict

def _get_ljung_box(residuals,num_param,out_dict):
    """
    Run a Ljung-Box test, interpret it, and append the results to out_dict. 
    
    Parameters
    ----------
    residuals : numpy.ndarray
        float array holding weighted fit residuals
    num_param : int
        number of fit parameters
    out_dict : dict
        dictionary with 'name', 'description', 'value', 'is_good', and 'msg' 
        keys pointing to list values

    Returns
    out_dict : dict
        out_dict updated with results of this quality test.
    """

    name = "ljung-box"
    txt = "Ljung-Box test for correlated residuals"
    value, _, _ = ljung_box(residuals,num_param=num_param)

    if value < 0.05:
        msg = f"A p-value of {value:.3e} for the Ljung-Box test is evidence for "
        msg += "correlated fit residuals. "
        msg += "This may mean your model does not describe your data well "
        msg += "(model violation). Look for regions where your model "
        msg += "systematically deviates from your data."
        is_good = False
    else:
        msg = f"A p-value of {value:.3e} for the Ljung-Box test "
        msg += "is consistent with the residuals having uncorrelated residuals."
        is_good = True

    out_dict["name"].append(name)
    out_dict["description"].append(txt)
    out_dict["value"].append(value)
    out_dict["is_good"].append(is_good)
    out_dict["message"].append(msg)

    return out_dict


def get_fit_quality(residuals,
                    num_param,
                    lnL,
                    success):
    """
    Do a collection of small analyses assessing fit quality. 
    
    Parameters
    ----------
    residuals : numpy.ndarray
        float array holding weighted fit residuals
    num_param : int
        number of fit parameters
    lnL : float
        log likelihood at fit parameter estimates
    success : bool
        whether or not the fit was successful, as determined by the Fitter
        object. 
    
    Returns
    -------
    out_df : pandas.DataFrame
        dataframe holding fit quality. columns are 'name' (test name); 
        'description' (test description), 'is_good' (whether or not the test
        indicates the fit is good), 'value' (value of the relevant test output),
        and 'msg' (string description of what the test output means)
    """

    num_obs = len(residuals)

    out_dict = {"name":[],
                "description":[],
                "is_good":[],
                "value":[],
                "message":[]}
    
    out_dict = _get_success(success=success,
                            out_dict=out_dict)

    out_dict = _get_num_obs(num_obs=num_obs,
                            out_dict=out_dict)
    
    out_dict = _get_num_param(num_param=num_param,
                              num_obs=num_obs,
                              out_dict=out_dict)

    out_dict = _get_lnL(lnL=lnL,
                        out_dict=out_dict)
    
    out_dict = _get_chi2(residuals=residuals,
                         num_param=num_param,
                         out_dict=out_dict)
    
    out_dict = _get_reduced_chi2(residuals=residuals,
                                 num_param=num_param,
                                 out_dict=out_dict)

    out_dict = _get_mean0_ttest(residuals=residuals,
                                out_dict=out_dict)

    out_dict = _get_durbin_watson(residuals=residuals,
                                  out_dict=out_dict)
    
    out_dict = _get_ljung_box(residuals=residuals,
                              num_param=num_param,
                              out_dict=out_dict)

    out_df = pd.DataFrame(out_dict)
    out_df.index = out_df["name"]
    out_df = out_df.drop(columns=["name"])

    return out_df