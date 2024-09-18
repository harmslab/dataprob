"""
Function to plot the fit results as a line through y_obs points.
"""

from dataprob.plot._plot_utils import get_plot_features
from dataprob.plot._plot_utils import get_style
from dataprob.plot._plot_utils import get_vectors

import matplotlib
from matplotlib import pyplot as plt

import numpy as np

def plot_fit(f,
             x_axis=None,
             x_label=None,
             y_label=None,
             num_samples=50,
             y_obs_style=None,
             y_std_style=None,
             y_calc_style=None,
             sample_line_style=None,
             legend=True,
             ax=None):
    """
    Plot the fit results as a line through y_obs points.

    Parameters
    ----------
    f : dataprob.Fitter
        dataprob.Fitter instance for which .fit() has been run
    x_axis : list-like, optional
        plot y_obs, y_std, and y_calc, against these x-axis values. If this
        is not specified, plot against 0 -> len(y_obs)-1
    x_label : str, optional
        label for the x-axis
    y_label : str, optional
        label for the y-axis
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
    sample_line_style : dict, optional
        matplotlib plot style keys to override the defaults for samples lines 
        plt.plot(**sample_line_style). 
    legend : bool, default=True
        add a legend to the plot
    ax : matplotlib.Axes, optional
        plot on a pre-defined axes

    Returns
    -------
    fig, ax : matplotlib.Figure, matplotlib.Axes
        matplotlib figure and axes on which the plot was done
    """

    # Validate inputs and prep for plotting
    has_results, x_label, y_label, num_samples = get_plot_features(f,
                                                                   x_label,
                                                                   y_label,
                                                                   num_samples)
    
    # Get styles for series
    y_obs_style = get_style(y_obs_style,"y_obs")
    y_std_style = get_style(y_std_style,"y_std")
    y_calc_style = get_style(y_calc_style,"y_calc")
    sample_line_style = get_style(sample_line_style,"sample_line")


    # Define plot axis
    if ax is None:
        fig, ax = plt.subplots(1,figsize=(6,6))

    if not issubclass(type(ax),matplotlib.axes.Axes):
        err = "ax should be a matplotlib Axes instance\n"
        raise ValueError(err)
    
    # Get values to plot
    x_axis, y_obs, y_std, y_calc = get_vectors(f,x_axis)
    
    good_mask = np.logical_not(np.isnan(y_obs))
    x_axis = x_axis[good_mask]
    y_obs = y_obs[good_mask]
    y_std = y_std[good_mask]
    y_calc = y_calc[good_mask]

    # Gracefully return if there are no observations to plot
    if len(y_obs) == 0:
        fig = ax.get_figure()
        return fig, ax

    # Create core plot
    ax.plot(x_axis,y_obs,**y_obs_style,label="y_obs")
    ax.errorbar(x=x_axis,y=y_obs,yerr=y_std,**y_std_style)
    ax.plot(x_axis,y_calc,**y_calc_style,label="y_calc")
    
    # Plot the samples
    if num_samples > 0:

        # Get a sample dataframe
        sample_df = f.get_sample_df(num_samples=num_samples)

        # Plot every sample column
        label = "samples"
        for c in sample_df.columns:
            
            # Skip y_obs, y_std, y_calc
            if c[0] == "y":
                continue

            # Grab sample
            s = sample_df.loc[:,c]

            # Plot sample    
            ax.plot(x_axis,s,label=label,**sample_line_style)

            # After the first loop, turn off sample label to keep legend sane
            if label == "samples":
                label = None

    # plot legend if requested
    if legend:
        ax.legend()

    # Set labels
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    
    fig = ax.get_figure()

    return fig, ax