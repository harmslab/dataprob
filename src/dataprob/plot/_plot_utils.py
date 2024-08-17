from dataprob.check import check_int

import numpy as np

import copy

def get_plot_features(f,
                      x_label,
                      y_label,
                      num_samples):
    """
    Get some generic information about plotting.

    Parameters
    ----------
    f : Fitter
        fitter for which fit() has successfully run
    x_label : str or None
        label for the x-axis. to omit, set to None
    y_label : str or None
        label for the y-axis. to omit, set to None
    num_samples : int, default=50
        number of samples to plot

    Returns
    -------
    x_label : str or None
        validated x_label
    y_label : str or None
        validated y_label
    num_samples : int
        validated number of samples 
    """
    
    # Make sure fit was successful before trying to plot anything
    if not f.success:
        err = "fit has not been successfully run. cannot generate plot.\n"
        raise RuntimeError(err)

    if x_label is not None:
        x_label = str(x_label)

    if y_label is not None:
        y_label = str(y_label)
    
    num_samples = check_int(value=num_samples,
                            variable_name="num_samples",
                            minimum_allowed=0)
    
    if f.samples is None:
        err = "could not get samples from this fit. cannot plot\n"
        raise ValueError(err)
    
    if len(f.samples) < num_samples:
        err = f"num_samples ({num_samples}) is more than the number of\n"
        err += f"samples in the fitter ({len(f.samples)})\n"
        raise ValueError(err)

    return x_label, y_label, num_samples

def validate_and_load_style(some_style,
                            some_style_name,
                            default_style):
    """
    Validate and load some style dictionary.
    
    Parameters
    ----------
    some_style : dict or None
        user-specified style dictionary
    some_style_name : str
        name of style dictionary for error message
    default_style : dict
        dictionary holding default style information
        
    Returns
    -------
    some_style : dict
        dictionary with final style
    """

    default_style = copy.deepcopy(default_style)
    
    if some_style is None:
        return default_style
        
    if not issubclass(type(some_style),dict):
        err = f"{some_style_name} should be a dictionary of matplotlib plot styles\n"
        raise ValueError(err)
    
    for k in some_style:
        default_style[k] = some_style[k]
    
    return default_style

def get_styling(y_obs_style=None,
                y_std_style=None,
                y_calc_style=None,
                sample_style=None):
    """
    Return styles to use for plotting features on the plot. Any key in these
    dictionaries overwrites keys in the default style, but does not alter other
    values in the default. 
    
    Parameters
    ----------
    y_obs_style : dict, optional
        styling dictionary for y_obs
    y_std_style : dict, optional
        styling dictionary for y_std
    y_calc_style : dict, optional
        styling dictionary for y_calc
    sample_style : dict, optional
        styling dictionary for samples
    
    Returns
    -------
    y_obs_style, y_std_style, y_calc_style, sample_style : dict
        dictionaries with appropriate styles assembled from defaults and user
        inputs
    """

    # In the future, these should probably be broken out into json or something
    # elsewhere in the code base... 

    # y_obs
    y_obs_style_default = {"marker":"o",
                           "markeredgecolor":"black",
                           "markerfacecolor":"none",
                           "markersize":4,
                           "lw":0,
                           "zorder":5}
    y_obs_style = validate_and_load_style(y_obs_style,
                                          "y_obs_style",
                                          y_obs_style_default)

    # y_std
    y_std_style_default = {"lw":0,
                           "ecolor":"black",
                           "elinewidth":1,
                           "capsize":3,
                           "zorder":4}
    y_std_style = validate_and_load_style(y_std_style,
                                          "y_std_style",
                                          y_std_style_default)

    # y_calc
    y_calc_style_default = {"linestyle":"-",
                            "color":"red",
                            "lw":2,
                            "zorder":10}
    y_calc_style = validate_and_load_style(y_calc_style,
                                           "y_calc_style",
                                           y_calc_style_default)

    # samples    
    sample_style_default = {"linestyle":"-",
                            "alpha":0.1,
                            "color":"gray",
                            "zorder":0}
    sample_style = validate_and_load_style(sample_style,
                                           "sample_style",
                                           sample_style_default)
    
    return y_obs_style, y_std_style, y_calc_style, sample_style

def get_vectors(f,x_axis=None):
    """
    Get a set of vectors, all the same length, describing results.
    
    Parameters
    ----------
    f : Fitter
        fitter object for which fit has been run
    x_axis : numpy.ndarray
        numpy array the same length as f.data_df["y_obs"] that should be used
        for the x-axis. if None, return a sequential array that is the right
        length

    Returns
    -------
    x_axis, y_obs, y_std, y_calc : numpy.ndarray
        numpy arrays for the four indicated features
    """

    # Grab y_obs, y_std, y_calc
    y_obs = np.array(f.data_df["y_obs"],dtype=float)
    y_std = np.array(f.data_df["y_std"],dtype=float)
    y_calc = np.array(f.data_df["y_calc"],dtype=float)
    
    # If no x-axis sent in, build one
    if x_axis is None:
        x_axis = np.arange(len(y_obs),dtype=float)

    if len(x_axis) != len(y_obs):
        err = "x_axis should be a numpy array the same length as y_obs\n"
        raise ValueError(err)

    return x_axis, y_obs, y_std, y_calc
    
def _get_edges(some_vector,scalar):
    """
    Get buffered edges above the min and max of values in some_vector.
    
    Parameters
    ----------
    some_vector : numpy array
        array of non-nan values to bracket
    scalar : float
        fractional expansion to apply

    Returns
    -------
    left : float
        value for left-most edge
    right : float
        value for right-most edge
    """

    total_span = np.max(some_vector) - np.min(some_vector)
    if total_span == 0:
        err = "minimum and maximum of vector must differ\n"
        raise ValueError(err)

    offset = total_span*scalar

    # Get left (or bottom) side
    left = np.min(some_vector) - offset
    
    # Get right (or top) side
    right = np.max(some_vector) + offset

    return left, right

def get_plot_dimensions(x_values,
                        y_values,
                        scalar=0.05):
    """
    Get the edges of a plot in matplotlib coordinates for drawing things based
    on their fractional position. 
    
    Parameters
    ----------
    x_values : numpy.ndarray
        numpy array of x_values that are being plotted
    y_values : numpy.ndarray
        numpy array of y_values that are being plotted
    scalar : float, default = 0.05
        expand beyond the edges of x and y by this fraction
    
    Returns
    -------
    x_left, x_right, y_bottom, y_top : float
        far left, far right, far bottom, far top values for plotting. 
    """
    
    x_left, x_right = _get_edges(x_values,scalar=scalar)
    y_bottom, y_top = _get_edges(y_values,scalar=scalar)
    
    return x_left, x_right, y_bottom, y_top