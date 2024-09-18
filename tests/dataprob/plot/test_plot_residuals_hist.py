
import pytest

import dataprob
from dataprob.fitters.ml import MLFitter

from dataprob.plot.plot_residuals_hist import plot_residuals_hist

import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import pandas as pd

def test_plot_residuals_hist():

    # covers whole decision tree, but not an amazing tests of outputs. 
    # graphical. 

    def test_fcn(m,b): return m*np.arange(100) + b
    y_obs = test_fcn(m=5,b=1)
    y_std = np.ones(100)*0.1

    f = MLFitter(some_function=test_fcn)

    # run on no-fit fitter. Should work fine, even though it has nothing to 
    # plot
    fig, ax = plot_residuals_hist(f)
    assert issubclass(type(fig),matplotlib.figure.Figure)
    assert issubclass(type(ax),matplotlib.axes.Axes)
    plt.close(fig)
    
    f = MLFitter(some_function=test_fcn)
    f.fit(y_obs=y_obs,
          y_std=y_std)
    
    fig, ax = plot_residuals_hist(f)
    assert issubclass(type(fig),matplotlib.figure.Figure)
    assert issubclass(type(ax),matplotlib.axes.Axes)
    plt.close(fig)


    fig, ax = plot_residuals_hist(f,plot_unweighted=True)
    assert issubclass(type(fig),matplotlib.figure.Figure)
    assert issubclass(type(ax),matplotlib.axes.Axes)
    plt.close(fig)
    
    with pytest.raises(ValueError):
        plot_residuals_hist(f,plot_unweighted="not_a_bool")

    fig, ax = plot_residuals_hist(f,x_label=None,y_label=None)
    assert issubclass(type(fig),matplotlib.figure.Figure)
    assert issubclass(type(ax),matplotlib.axes.Axes)
    plt.close(fig)

    # This will raise am attribute error if this is actually eventually passing
    # to matplotlib.pyplot.fill because it's not a real style. nice (but implicit)
    # test
    with pytest.raises(AttributeError):
        fig, ax = plot_residuals_hist(f,hist_bar_style={"not_real_style":10})

    # But this should now work
    fig, ax = plot_residuals_hist(f,hist_bar_style={"edgecolor":"pink"})
    assert issubclass(type(fig),matplotlib.figure.Figure)
    assert issubclass(type(ax),matplotlib.axes.Axes)
    plt.close(fig)

    with pytest.raises(ValueError):
        plot_residuals_hist(f,hist_bar_style="not_a_dict")
    
    fig, ax = plt.subplots(1,figsize=(6,6))
    fig_out, ax_out = plot_residuals_hist(f,ax=ax)
    assert fig_out is fig
    assert ax_out is ax
    plt.close(fig)

    with pytest.raises(ValueError):
        plot_residuals_hist(f,ax="not_an_ax")

    fig, ax = plot_residuals_hist(f,plot_unweighted=False)
    assert issubclass(type(fig),matplotlib.figure.Figure)
    assert issubclass(type(ax),matplotlib.axes.Axes)
    plt.close(fig)

    fig, ax = plot_residuals_hist(f,plot_unweighted=True)
    assert issubclass(type(fig),matplotlib.figure.Figure)
    assert issubclass(type(ax),matplotlib.axes.Axes)
    plt.close(fig)

    # amke sure we can send in hist bins
    fig, ax = plot_residuals_hist(f,hist_bins=20)
    assert issubclass(type(fig),matplotlib.figure.Figure)
    assert issubclass(type(ax),matplotlib.axes.Axes)
    plt.close(fig)

    with pytest.raises(ValueError):
        plot_residuals_hist(f,hist_bins="stupid_bins")

    # Send in fitter without unweighted_residuals. It should not crash 
    def test_fcn(m,b): return m*np.arange(100) + b
    f = dataprob.setup(some_function=test_fcn)
    f.data_df = pd.DataFrame({"y_obs":test_fcn(m=5,b=1),
                              "y_std":np.ones(100)*0.1})
    f.param_df.loc["m","guess"] = np.nan
    assert np.array_equal(f.data_df.columns,["y_obs","y_std"])
    fig, ax = plot_residuals_hist(f,plot_unweighted=True)    
    assert issubclass(type(fig),matplotlib.figure.Figure)
    assert issubclass(type(ax),matplotlib.axes.Axes)

    # Send in fitter without residuals. It should work fine. 
    def test_fcn(m,b): return m*np.arange(100) + b
    f = dataprob.setup(some_function=test_fcn)
    f.data_df = pd.DataFrame({"y_obs":test_fcn(m=5,b=1),
                              "y_std":np.ones(100)*0.1})
    f._y_std = None # <- nuke _y_std
    assert np.array_equal(f.data_df.columns,["y_obs","y_calc","unweighted_residuals"])

    fig, ax = plot_residuals_hist(f,plot_unweighted=False)    
    assert issubclass(type(fig),matplotlib.figure.Figure)
    assert issubclass(type(ax),matplotlib.axes.Axes)
