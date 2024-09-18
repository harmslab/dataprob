"""
Function to plot a histogram of residuals. 
"""

from dataprob.util.check import check_bool
from dataprob.plot._plot_utils import get_plot_dimensions
from dataprob.plot._plot_utils import get_plot_features
from dataprob.plot._plot_utils import get_style

import matplotlib
from matplotlib import pyplot as plt
import numpy as np

def plot_residuals_hist(f,
                        x_label=None,
                        y_label=None,
                        hist_bins=None,
                        y_calc_style=None,
                        hist_bar_style=None,
                        plot_unweighted=False,
                        ax=None):
    """
    Plot a histogram of residuals. 

    Parameters
    ----------
    f : dataprob.Fitter
        dataprob.Fitter instance for which .fit() has been run
    x_label : str, default="x"
        label for the x-axis. to omit, set to None
    y_label : str, default="y"
        label for the y-axis. to omit, set to None
    hist_bins : bool, optional  
        customize bins in histogram. if specified, this will be passed directly
        to numpy.histogram via the "bins" argument. 
    y_calc_style : dict, optional
        matplotlib plot style here to override the defaults for y_calc. Used via
        plt.plot(**y_calc_style). 
    hist_bar_style : dict, optional
        matplotlib plot style keys here to override the defaults for the
        histogram bars. Used via plt.fill(**hist_bar_style). 
    plot_unweighted : bool, False
        plot unweighted (rather than weighted) residuals
    ax : matplotlib.Axes, optional
        plot on the pre-defined axes

    Returns
    -------
    fig, ax : matplotib.Figure, matplotlib.Axes
        matplotlib figure and axes on which the plot was done
    """

    # Validate inputs and prep for plotting
    _, x_label, y_label, _ = get_plot_features(f,
                                               x_label,
                                               y_label,
                                               num_samples=0)
    
    plot_unweighted = check_bool(value=plot_unweighted,
                                 variable_name="plot_unweighted")
    
    y_calc_style = get_style(user_style=y_calc_style,
                             style_name="y_calc")
    hist_bar_style = get_style(user_style=hist_bar_style,
                               style_name="hist_bar")
    
    # -----------------------------------------------------------------------
    # Generate plot

    # create ax
    if ax is None:
        fig, ax = plt.subplots(1,figsize=(6,6))

    if not issubclass(type(ax),matplotlib.axes.Axes):
        err = "ax should be a matplotlib Axes instance\n"
        raise ValueError(err)

    # Figure out what to plot
    if plot_unweighted:

        # Gracefully fail if no unweighted residuals in fitter
        if "unweighted_residuals" not in f.data_df.columns:
            fig = ax.get_figure()
            return fig, ax

        residual = f.data_df["unweighted_residuals"]
    else:

        # Gracefully fail if no weighted residuals in fitter
        if "weighted_residuals" not in f.data_df.columns:
            fig = ax.get_figure()
            return fig, ax

        residual = f.data_df["weighted_residuals"]

    # Generate histogram
    if hist_bins is not None:
        counts, edges = np.histogram(residual,bins=hist_bins)
    else:
        counts, edges = np.histogram(residual)

    # Plot histogram bars
    for i in range(len(counts)):
        
        box_x = [edges[i],edges[i],edges[i+1],edges[i+1]]
        box_y = [0,counts[i],counts[i],0]
        
        ax.fill(box_x,box_y,**hist_bar_style)
        
    # Get positioning information
    x_left, x_right, y_bottom, y_top = get_plot_dimensions(edges,counts)

    # Plot mean
    m = np.mean(residual)
    ax.plot([m,m],[y_bottom,y_top],**y_calc_style)
    ax.plot([0,0],[y_bottom,y_top],'--',color="gray",zorder=6)

    # Set axis labels
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
            
    fig = ax.get_figure()

    return fig, ax