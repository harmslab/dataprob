"""
Utility functions used internally by the plotting functions to calculate things
like the plot size, check variable types, etc. 
"""

from dataprob.util.check import check_int
from dataprob.plot import appearance

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
    
    # Check for plot success
    has_fit_results = f.success is True
    
    if x_label is not None:
        x_label = str(x_label)

    if y_label is not None:
        y_label = str(y_label)
    
    num_samples = check_int(value=num_samples,
                            variable_name="num_samples",
                            minimum_allowed=0)
    
    if f.samples is None:
        num_samples = 0
    
    if f.samples is not None and len(f.samples) < num_samples:
        num_samples = len(f.samples)

    return has_fit_results, x_label, y_label, num_samples

def get_style(user_style,style_name):
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

    if style_name not in appearance.default_styles:
        err = f"style_name {style_name} was not found in appearance.default_styles\n"
        err += "It should be one of:\n"
        for k in appearance.default_styles:
            err += f"    {k}\n"
        raise ValueError(err)

    output_style = copy.deepcopy(appearance.default_styles[style_name])
    
    if user_style is None:
        return output_style
        
    if not issubclass(type(user_style),dict):
        err = f"{user_style} should be a dictionary of matplotlib plot styles\n"
        raise ValueError(err)
    
    for k in user_style:
        output_style[k] = user_style[k]
    
    return output_style


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
    if "y_obs" in f.data_df.columns:
        y_obs = np.array(f.data_df["y_obs"],dtype=float).copy()
    else:
        y_obs = np.ones(len(f.data_df))*np.nan
    
    if "y_std" in f.data_df.columns:
        y_std = np.array(f.data_df["y_std"],dtype=float).copy()
    else:
        y_std = np.ones(len(y_obs))*np.nan

    if "y_calc" in f.data_df.columns:
        y_calc = np.array(f.data_df["y_calc"],dtype=float).copy()
    else:
        y_calc = np.ones(len(y_obs))*np.nan

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


def sync_axes(ax1,ax2,sync_axis):
    """
    Synchronize the limits of two matplotlib.Axes objects. Sets to maximum 
    extent that covers both current axes. Update the axis objects in place. 
    
    Parameters
    ----------
    ax1, ax2: matplotlib.Axes, matplotlib.Axes
        axes objects to sync
    sync_axis : str
        'x' or 'y' -- which axis to synchronize between. 

    Returns
    -------
    None
    """

    getter = f"get_{sync_axis}lim"
    lim1 = ax1.__getattribute__(getter)()
    lim2 = ax2.__getattribute__(getter)()

    new_lim = [np.min([lim1[0],lim2[0]]),
                np.max([lim1[1],lim2[1]])]

    setter = f"set_{sync_axis}lim"
    ax1.__getattribute__(setter)(new_lim)
    ax2.__getattribute__(setter)(new_lim)