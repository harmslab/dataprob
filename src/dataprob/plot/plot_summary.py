"""
Function to generate summary plot for fit.
"""

from dataprob.plot.plot_fit import plot_fit
from dataprob.plot.plot_residuals import plot_residuals
from dataprob.plot.plot_residuals_hist import plot_residuals_hist
from dataprob.plot._plot_utils import get_vectors
from dataprob.plot._plot_utils import sync_axes

from matplotlib import pyplot as plt

def plot_summary(f,
                 x_axis=None,
                 x_label=None,
                 y_label=None,
                 num_samples=50,                     
                 y_obs_style=None,
                 y_std_style=None,
                 y_calc_style=None,
                 sample_line_style=None,
                 sample_point_style=None,
                 hist_bar_style=None,
                 plot_unweighted=False):
    """
    Create a fit summary plot showing the main fit and residuals. 
    
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
    sample_point_style : dict, optional
        matplotlib plot style keys to override the defaults for samples points 
        plt.plot(**sample_point_style). 
    hist_bar_style : dict, optional
        matplotlib plot style keys here to override the defaults for the
        histogram bars. Used via plt.fill(**hist_bar_style). 
    plot_unweighted : bool, default=False
        plot unweighted (rather than weighted) residuals

    Returns
    -------
    fig : matplotlib.figure.Figure
        matplotlib Figure holding plot (with four gridspec axes)
    """

    
    gs_kw = dict(width_ratios=[4,1],
                 height_ratios=[4,1])
    
    fig, axd = plt.subplot_mosaic([['main_plot', 'y_residuals'],
                                   ['x_residuals', 'residuals_hist']],
                                  gridspec_kw=gs_kw,
                                  figsize=(7.5, 7.5),
                                  layout="tight")
    
    plot_fit(f,
             x_axis=x_axis,
             num_samples=num_samples,
             y_obs_style=y_obs_style,
             y_std_style=y_std_style,
             y_calc_style=y_calc_style,
             sample_line_style=sample_line_style,
             legend=False,
             ax=axd["main_plot"])
    
    plot_residuals(f,
                   x_axis=x_axis,
                   num_samples=num_samples,
                   y_obs_style=y_obs_style,
                   y_std_style=y_std_style,
                   y_calc_style=y_calc_style,
                   sample_point_style=sample_point_style,
                   plot_y_residuals=False,
                   plot_unweighted=plot_unweighted,
                   ax=axd["x_residuals"])
    
    plot_residuals(f,
                   num_samples=num_samples,
                   y_obs_style=y_obs_style,
                   y_std_style=y_std_style,
                   y_calc_style=y_calc_style,
                   sample_point_style=sample_point_style,
                   plot_y_residuals=True,
                   plot_unweighted=plot_unweighted,
                   ax=axd["y_residuals"])
    
    plot_residuals_hist(f,
                        y_calc_style=y_calc_style,
                        hist_bar_style=hist_bar_style,
                        plot_unweighted=plot_unweighted,
                        ax=axd["residuals_hist"])
    
    # sync limits for x-axis of main plot and x residuals
    sync_axes(axd["main_plot"],
              axd["x_residuals"],
              sync_axis="x")
    
    # sync limits for y-axis of main plot and y residuals
    sync_axes(axd["main_plot"],
              axd["y_residuals"],
              sync_axis="y")
    
    # sync limits for x-axis of residuals histogram and y residuals
    sync_axes(axd["residuals_hist"],
              axd["y_residuals"],
              sync_axis="x")
    
    # Clean up axes and spines
    axd["main_plot"].get_xaxis().set_visible(False)
    axd["main_plot"].spines['right'].set_visible(False)
    axd["main_plot"].spines['top'].set_visible(False)
    axd["main_plot"].spines['bottom'].set_visible(False)
    axd["main_plot"].set_ylabel(y_label)
    
    axd["x_residuals"].spines['right'].set_visible(False)
    axd["x_residuals"].spines['top'].set_visible(False)
    axd["x_residuals"].set_xlabel(x_label)
    axd["x_residuals"].set_ylabel("residual")
    
    axd["y_residuals"].set_axis_off()
    
    axd["residuals_hist"].get_yaxis().set_visible(False)
    axd["residuals_hist"].spines['right'].set_visible(False)
    axd["residuals_hist"].spines['top'].set_visible(False)
    axd["residuals_hist"].spines['left'].set_visible(False)
    axd["residuals_hist"].set_xlabel("residual")

    fig.tight_layout() 

    return fig   
    