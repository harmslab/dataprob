
import pytest

import dataprob
from dataprob.fitters.ml import MLFitter

from dataprob.plot.plot_residuals import plot_residuals

import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import pandas as pd

def test_plot_residuals():

    # covers whole decision tree, but not an amazing tests of outputs. 
    # graphical. 

    def test_fcn(m,b): return m*np.arange(10) + b
    y_obs = test_fcn(m=5,b=1)
    y_std = np.ones(10)*0.1

    f = MLFitter(some_function=test_fcn)
    
    # run on no-fit fitter. Should fail gracefully. 
    fig, ax = plot_residuals(f)
    assert issubclass(type(fig),matplotlib.figure.Figure)
    assert issubclass(type(ax),matplotlib.axes.Axes)
    plt.close(fig)
    
    f = MLFitter(some_function=test_fcn)
    f.fit(y_obs=y_obs,
          y_std=y_std)
    
    fig, ax = plot_residuals(f)
    assert issubclass(type(fig),matplotlib.figure.Figure)
    assert issubclass(type(ax),matplotlib.axes.Axes)
    plt.close(fig)

    with pytest.raises(ValueError):
        plot_residuals(f,plot_unweighted="not_a_bool")

    fig, ax = plot_residuals(f,plot_unweighted=True)
    assert issubclass(type(fig),matplotlib.figure.Figure)
    assert issubclass(type(ax),matplotlib.axes.Axes)
    plt.close(fig)
    
    with pytest.raises(ValueError):
        plot_residuals(f,plot_y_residuals="not_a_bool")

    fig, ax = plot_residuals(f,
                             plot_unweighted=True,
                             plot_y_residuals=True,
                             num_samples=10)
    assert issubclass(type(fig),matplotlib.figure.Figure)
    assert issubclass(type(ax),matplotlib.axes.Axes)
    plt.close(fig)

    fig, ax = plot_residuals(f,
                             plot_unweighted=True,
                             plot_y_residuals=True,
                             num_samples=0)
    assert issubclass(type(fig),matplotlib.figure.Figure)
    assert issubclass(type(ax),matplotlib.axes.Axes)
    plt.close(fig)

    fig, ax = plot_residuals(f,
                             plot_unweighted=True,
                             plot_y_residuals=False,
                             num_samples=10)
    assert issubclass(type(fig),matplotlib.figure.Figure)
    assert issubclass(type(ax),matplotlib.axes.Axes)
    plt.close(fig)

    fig, ax = plot_residuals(f,
                             plot_unweighted=True,
                             plot_y_residuals=False,
                             num_samples=0)
    assert issubclass(type(fig),matplotlib.figure.Figure)
    assert issubclass(type(ax),matplotlib.axes.Axes)
    plt.close(fig)

    fig, ax = plot_residuals(f,
                             plot_unweighted=False,
                             plot_y_residuals=True,
                             num_samples=10)
    assert issubclass(type(fig),matplotlib.figure.Figure)
    assert issubclass(type(ax),matplotlib.axes.Axes)
    plt.close(fig)

    fig, ax = plot_residuals(f,
                             plot_unweighted=False,
                             plot_y_residuals=True,
                             num_samples=0)
    assert issubclass(type(fig),matplotlib.figure.Figure)
    assert issubclass(type(ax),matplotlib.axes.Axes)
    plt.close(fig)

    fig, ax = plot_residuals(f,
                             plot_unweighted=False,
                             plot_y_residuals=False,
                             num_samples=10)
    assert issubclass(type(fig),matplotlib.figure.Figure)
    assert issubclass(type(ax),matplotlib.axes.Axes)
    plt.close(fig)

    fig, ax = plot_residuals(f,
                             plot_unweighted=False,
                             plot_y_residuals=False,
                             num_samples=0)
    assert issubclass(type(fig),matplotlib.figure.Figure)
    assert issubclass(type(ax),matplotlib.axes.Axes)
    plt.close(fig)

    fig, ax = plt.subplots(1,figsize=(6,6))
    fig_out, ax_out = plot_residuals(f,ax=ax)
    assert fig_out is fig
    assert ax_out is ax
    plt.close(fig)

    # bad ax pass
    with pytest.raises(ValueError):
        plot_residuals(f,ax="not_an_ax")

    # stick a nan into one of the samples -- should work fine
    f._samples[0,:] = np.nan
    fig, ax = plt.subplots(1,figsize=(6,6))
    fig_out, ax_out = plot_residuals(f,ax=ax)
    assert fig_out is fig
    assert ax_out is ax
    plt.close(fig)


    # Send in fitter without unweighted_residuals. It should not crash 
    def test_fcn(m,b): return m*np.arange(100) + b
    f = dataprob.setup(some_function=test_fcn)
    f.data_df = pd.DataFrame({"y_obs":test_fcn(m=5,b=1),
                              "y_std":np.ones(100)*0.1})
    f.param_df.loc["m","guess"] = np.nan
    assert np.array_equal(f.data_df.columns,["y_obs","y_std"])
    fig, ax = plot_residuals(f,plot_unweighted=True)    
    assert issubclass(type(fig),matplotlib.figure.Figure)
    assert issubclass(type(ax),matplotlib.axes.Axes)

    # Send in fitter without residuals. It should work fine. 
    def test_fcn(m,b): return m*np.arange(100) + b
    f = dataprob.setup(some_function=test_fcn)
    f.data_df = pd.DataFrame({"y_obs":test_fcn(m=5,b=1),
                              "y_std":np.ones(100)*0.1})
    f._y_std = None # <- nuke _y_std
    assert np.array_equal(f.data_df.columns,["y_obs","y_calc","unweighted_residuals"])

    fig, ax = plot_residuals(f,plot_unweighted=False)    
    assert issubclass(type(fig),matplotlib.figure.Figure)
    assert issubclass(type(ax),matplotlib.axes.Axes)




    