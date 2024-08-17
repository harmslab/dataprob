
from dataprob.plot._plot_utils import get_plot_features
from dataprob.plot._plot_utils import get_styling
from dataprob.plot._plot_utils import get_vectors

import matplotlib
from matplotlib import pyplot as plt

def plot_fit(f,
             x_axis=None,
             x_label="x",
             y_label="y",
             num_samples=50,
             y_obs_style=None,
             y_std_style=None,
             y_calc_style=None,
             sample_style=None,
             ax=None):
    """
    Plot the fit results as a line through y_obs points.

    Parameters
    ----------
    f : dataprob.Fitter
        dataprob.Fitter instance for which .fit() has been run
    x_axis : list-like, optional
        plot y_obs, y_std, and y_calc, against these x-axis values. If this
        is not specified, plot against [0,len(y_obs)-1]
    x_label : str, default="x"
        label for the x-axis. to omit, set to None
    y_label : str, default="y"
        label for the y-axis. to omit, set to None
    num_samples : int, default=50
        number of samples to plot
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
    
    y_obs_style, y_std_style, y_calc_style, sample_style = get_styling(y_obs_style,
                                                                       y_std_style,
                                                                       y_calc_style,
                                                                       sample_style)

    x_axis, y_obs, y_std, y_calc = get_vectors(f,x_axis)

    
    # -----------------------------------------------------------------------
    # Generate plot

    if ax is None:
        fig, ax = plt.subplots(1,figsize=(6,6))

    if not issubclass(type(ax),matplotlib.axes.Axes):
        err = "ax should be a matplotlib Axes instance\n"
        raise ValueError(err)
    
    ax.plot(x_axis,y_obs,**y_obs_style,label="y_obs")
    ax.errorbar(x=x_axis,y=y_obs,yerr=y_std,**y_std_style)
    ax.plot(x_axis,y_calc,**y_calc_style,label="y_calc")
    
    if num_samples > 0:
        sample_df = f.get_sample_df(num_samples=num_samples)
        for i in range(num_samples):
            
            s = sample_df.loc[:,sample_df.columns[i+3]]
            
            if i == 0:
                label = "samples"
            else:
                label = None
                
            plt.plot(x_axis,s,label=label,**sample_style)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend()
    
    fig = ax.get_figure()

    return fig, ax