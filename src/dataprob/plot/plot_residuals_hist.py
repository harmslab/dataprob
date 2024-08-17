
from dataprob.check import check_bool
from dataprob.plot._plot_utils import get_plot_dimensions

import matplotlib
from matplotlib import pyplot as plt
import numpy as np
from scipy import stats

def plot_residuals_hist(f,
                        plot_unweighted=False,
                        x_label="y_calc - y_obs",
                        y_label="counts",
                        bar_style=None,
                        stats_as_text=True,
                        ax=None):
    """
    Plot the fit results as a line through y_obs points.

    Parameters
    ----------
    f : dataprob.Fitter
        dataprob.Fitter instance for which .fit() has been run
    plot_unweighted : bool, False
        plot unweighted (rather than weighted) residuals
    x_label : str, default="x"
        label for the x-axis. to omit, set to None
    y_label : str, default="y"
        label for the y-axis. to omit, set to None
    bar_style : dict, optional
        set matplotlib plot style keys here to override the defaults for the
        histogram bars. sent to plt.fill
    stats_as_text : bool, default=True
        write statistics about distribution as text on the plot
    ax : matplotlib.Axes, optional
        plot on the pre-defined axes

    Returns
    -------
    fig, ax : matplotib.Figure, matplotlib.Axes
        matplotlib figure and axes on which the plot was done
    """

    # -----------------------------------------------------------------------
    # Parse formatting inputs

    # Make sure fit was successful before trying to plot anything
    if not f.success:
        err = "fit has not been successfully run. cannot generate plot.\n"
        raise RuntimeError(err)
    
    plot_unweighted = check_bool(value=plot_unweighted,
                                 variable_name="plot_unweighted")

    if x_label is not None:
        x_label = str(x_label)

    if y_label is not None:
        y_label = str(y_label)
    
    if bar_style is None:
        bar_style = {}
    if not issubclass(type(bar_style),dict):
        err = "bar_style should be a dictionary of matplotlib plot styles\n"
        raise ValueError(err)

    _bar_style = {"lw":1,
                  "edgecolor":"black",
                  "facecolor":"gray"}
    for k in bar_style:
        _bar_style[k] = bar_style[k]
        
    
    stats_as_text = check_bool(value=stats_as_text,
                               variable_name="stats_as_text")

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
        residual = f.data_df["unweighted_residuals"]
    else:
        residual = f.data_df["weighted_residuals"]

    # Generate histogram
    counts, edges = np.histogram(residual)

    # Plot histogram bars
    for i in range(len(counts)):
        
        box_x = [edges[i],edges[i],edges[i+1],edges[i+1]]
        box_y = [0,counts[i],counts[i],0]
        
        ax.fill(box_x,box_y,**_bar_style)
        
    # Get positioning information
    x_left, x_right, _, y_top = get_plot_dimensions(edges,counts)
    x_span = x_right - x_left

    # Plot mean
    m = np.mean(residual)
    plt.plot([m,m],[0,y_top],"--",lw=2,color="red")

    # Set axis labels
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    
    # If the user requested stats
    if stats_as_text:

        # Do test
        is_normal = stats.normaltest(residual)

        # Figure out p-value formating
        p_value = is_normal.pvalue
        if p_value > 0.001:
            p_str = f"p: {p_value:7.3f}"
        else:
            p_str = f"p: {p_value:7.3e}"

        # Write mean
        m_str = f"{m:.3f}"
        ax.text(x_left + 0.05*x_span,y_top*0.97,"mean",zorder=20,ha="left")
        ax.text(x_left + 0.05*x_span,y_top*0.92,m_str,zorder=20,ha="left")
        
        # Write normality test p-value
        ax.text(x_left + 0.95*x_span,y_top*0.97,"Differs from normal?",zorder=20,ha="right")
        ax.text(x_left + 0.95*x_span,y_top*0.92,p_str,zorder=20,ha="right")
        
    fig = ax.get_figure()

    return fig, ax