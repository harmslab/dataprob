"""
Function to plot the fit residuals.
"""

from dataprob.plot._plot_utils import get_plot_features
from dataprob.plot._plot_utils import get_style
from dataprob.plot._plot_utils import get_vectors
from dataprob.plot._plot_utils import get_plot_dimensions
from dataprob.check import check_bool

import matplotlib
from matplotlib import pyplot as plt
import numpy as np

def plot_residuals(f,
                   x_axis=None,
                   x_label=None,
                   y_label=None,
                   num_samples=50,
                   y_obs_style=None,
                   y_std_style=None,
                   y_calc_style=None,
                   sample_point_style=None,
                   plot_y_residuals=False,
                   plot_unweighted=False,
                   ax=None):
    """
    Plot the fit residuals. 

    Parameters
    ----------
    f : dataprob.Fitter
        dataprob.Fitter instance for which .fit() has been run
    x_axis : list-like, optional
        plot y_obs, y_std, and y_calc, against these x-axis values. If this
        is not specified, plot against 0 -> len(y_obs)-1
    x_label : str, default="x"
        label for the x-axis. to omit, set to None
    y_label : str, default="y"
        label for the y-axis. to omit, set to None
    num_samples : int, default=50
        number of samples to plot. To not plot samples, set to 0. 
    y_obs_style : dict, optional
        matplotlib plot style keys to override the defaults for y_obs. Used via
        plt.plot(**y_obs_style). 
    y_std_style : dict, optional
        matplotlib plot style keys to override the defaults for y_std. Used via
        plt.errorbar(**y_std_style). 
    y_calc_style : dict, optional
        matplotlib plot style here to override the defaults for y_calc. Used via
        plt.plot(**y_calc_style). 
    sample_point_style : dict, optional
        matplotlib plot style keys to override the defaults for samples points 
        plt.plot(**sample_point_style). 
    plot_y_residuals : bool, default=False
        the default plots residual vs. x-axis. if True, this plots y-obs vs. 
        residual
    plot_unweighted : bool, default=False
        plot unweighted (rather than weighted) residuals
    ax : matplotlib.Axes, optional
        plot on the pre-defined axes

    Returns
    -------
    fig, ax : matplotib.Figure, matplotlib.Axes
        matplotlib figure and axes on which the plot was done
    """

    x_label, y_label, num_samples = get_plot_features(f,
                                                      x_label,
                                                      y_label,
                                                      num_samples)
    
    # Get styles for series
    y_obs_style = get_style(y_obs_style,"y_obs")
    y_std_style = get_style(y_std_style,"y_std")
    y_calc_style = get_style(y_calc_style,"y_calc")
    sample_point_style = get_style(sample_point_style,"sample_point")
    
    x_axis, y_obs, y_std, _ = get_vectors(f,x_axis=x_axis)

    plot_y_residuals = check_bool(value=plot_y_residuals,
                                  variable_name="plot_y_residuals")
    
    plot_unweighted = check_bool(value=plot_unweighted,
                                 variable_name="plot_unweighted")

    # Get residual samples
    if num_samples > 0:

        sample_df = f.get_sample_df(num_samples=num_samples)
        
        if plot_unweighted:
            denominator = 1
        else:
            denominator = y_std

        for c in sample_df.columns:
            if c[0] == "y": continue # skip y_obs, etc. 
            sample_df[c] = (sample_df[c] - y_obs)/denominator

    
    # -----------------------------------------------------------------------
    # Generate plot

    if ax is None:
        fig, ax = plt.subplots(1,figsize=(6,6))

    if not issubclass(type(ax),matplotlib.axes.Axes):
        err = "ax should be a matplotlib Axes instance\n"
        raise ValueError(err)

    if plot_unweighted:
        residual = f.data_df["unweighted_residuals"]
    else:
        residual = f.data_df["weighted_residuals"]

    x_left, x_right, y_bottom, y_top = get_plot_dimensions(x_axis,y_obs)
    mean_r = np.mean(residual)

    if plot_y_residuals:

        main_x = residual
        main_y = y_obs
        mean_x = [mean_r,mean_r]
        mean_y = [y_bottom,y_top]
        sample_is_x = True

    else:

        main_x = x_axis
        main_y = residual
        mean_x = [x_left,x_right]
        mean_y = [mean_r,mean_r]
        sample_is_x = False
        

    ax.plot(main_x,main_y,**y_obs_style)
    ax.errorbar(x=main_x,
                y=main_y,
                yerr=y_std,
                **y_std_style)
    ax.plot(mean_x,mean_y,**y_calc_style)

    if num_samples > 0:
        
        for c in sample_df.columns:

            if c[0] == "y": continue # skip y_obs, etc. 
            
            if sample_is_x:
                ax.plot(sample_df[c],main_y,**sample_point_style)
            else:
                ax.plot(main_x,sample_df[c],**sample_point_style) 

    
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    
    fig = ax.get_figure()

    return fig, ax
    