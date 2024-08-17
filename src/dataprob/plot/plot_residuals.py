
from dataprob.plot._plot_utils import get_plot_features
from dataprob.plot._plot_utils import get_styling
from dataprob.plot._plot_utils import get_vectors
from dataprob.plot._plot_utils import validate_and_load_style
from dataprob.check import check_bool

import matplotlib
from matplotlib import pyplot as plt
import numpy as np

def plot_residuals(f,
                   is_right_side=False,
                   x_axis=None,
                   x_label=None,
                   y_label=None,
                   num_samples=50,
                   y_obs_style=None,
                   y_std_style=None,
                   y_calc_style=None,
                   sample_style=None,
                   plot_unweighted=False,
                   ax=None):
    """
    Plot the fit residuals.

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
    y_obs_style : dict, optional
        set matplotlib plot style keys here to override the defaults for y_obs.
        sent to plt.plot
    y_std_style : dict, optional
        set matplotlib plot style keys here to override the defaults for y_std.
        sent to plt.errorbar
    y_calc_style : dict, optional
        set matplotlib plot style keys here to override the defaults for y_calc.
        sent to plt.plot
    sample_style : dict, optional
        set matplotlib plot style keys here to override the defaults for samples.
        sent to plt.plot
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
    
    y_obs_style, y_std_style, y_calc_style, _ = get_styling(y_obs_style,
                                                            y_std_style,
                                                            y_calc_style,
                                                            None)
    
    # default sample style should be points for this kind of plot. Validate 
    # independently from the normal "get_styling call"
    sample_style_default = {"marker":"o",
                            "alpha":0.1,
                            "markeredgecolor":"black",
                            "markerfacecolor":"gray",
                            "linewidth":0,
                            "zorder":0}

    sample_style = validate_and_load_style(some_style=sample_style,
                                           some_style_name="sample_style",
                                           default_style=sample_style_default)
    

    x_axis, y_obs, y_std, _ = get_vectors(f,x_axis=x_axis)

    plot_unweighted = check_bool(value=plot_unweighted,
                                 variable_name="plot_unweighted")

    is_right_side = check_bool(value=is_right_side,
                               variable_name="is_right_side")
    

    if num_samples > 0:

        sample_df = f.get_sample_df(num_samples=num_samples)
        
        if plot_unweighted:
            denominator = 1
        else:
            denominator = y_std

        for c in sample_df.columns[3:]:
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

    if is_right_side:

        ax.plot(residual,y_obs,**y_obs_style,label="y_obs")
        ax.errorbar(x=residual,y=y_obs,yerr=y_std,**y_std_style)
        
        m = np.mean(f.data_df["weighted_residuals"])
        ax.plot(m*np.ones(y_obs.shape[0]),
                f.data_df["y_obs"],'--',lw=1,color='gray',zorder=0)
        
        if num_samples > 0:
            for c in sample_df.columns[3:]:
                ax.plot(sample_df[c],y_obs,**sample_style)
        
    else:
        ax.plot(x_axis,residual,**y_obs_style,label="y_obs")
        ax.errorbar(x=x_axis,y=residual,yerr=y_std,**y_std_style)
        m = np.mean(f.data_df["weighted_residuals"])
        ax.plot(x_axis,
                m*np.ones(y_obs.shape[0]),'--',lw=1,color='gray',zorder=0)
        
        if num_samples > 0:
            for c in sample_df.columns[3:]:
                ax.plot(x_axis,sample_df[c],**sample_style)
    
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    
    fig = ax.get_figure()

    return fig, ax
    