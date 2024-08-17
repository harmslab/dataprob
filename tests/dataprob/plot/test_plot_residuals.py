
import pytest

from dataprob.fitters.ml import MLFitter

from dataprob.plot.plot_residuals import plot_residuals

import numpy as np
import matplotlib
from matplotlib import pyplot as plt

def test_plot_residuals():

    # covers whole decision tree, but not an amazing tests of outputs. 
    # graphical. 

    def test_fcn(m,b): return m*np.arange(10) + b
    y_obs = test_fcn(m=5,b=1)
    y_std = np.ones(10)*0.1

    f = MLFitter(some_function=test_fcn)

    # run on no-fit fitter
    with pytest.raises(RuntimeError):
        plot_residuals(f)
    
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
        plot_residuals(f,is_right_side="not_a_bool")

    fig, ax = plot_residuals(f,
                             plot_unweighted=True,
                             is_right_side=True,
                             num_samples=10)
    assert issubclass(type(fig),matplotlib.figure.Figure)
    assert issubclass(type(ax),matplotlib.axes.Axes)
    plt.close(fig)

    fig, ax = plot_residuals(f,
                             plot_unweighted=True,
                             is_right_side=True,
                             num_samples=0)
    assert issubclass(type(fig),matplotlib.figure.Figure)
    assert issubclass(type(ax),matplotlib.axes.Axes)
    plt.close(fig)

    fig, ax = plot_residuals(f,
                             plot_unweighted=True,
                             is_right_side=False,
                             num_samples=10)
    assert issubclass(type(fig),matplotlib.figure.Figure)
    assert issubclass(type(ax),matplotlib.axes.Axes)
    plt.close(fig)

    fig, ax = plot_residuals(f,
                             plot_unweighted=True,
                             is_right_side=False,
                             num_samples=0)
    assert issubclass(type(fig),matplotlib.figure.Figure)
    assert issubclass(type(ax),matplotlib.axes.Axes)
    plt.close(fig)

    fig, ax = plot_residuals(f,
                             plot_unweighted=False,
                             is_right_side=True,
                             num_samples=10)
    assert issubclass(type(fig),matplotlib.figure.Figure)
    assert issubclass(type(ax),matplotlib.axes.Axes)
    plt.close(fig)

    fig, ax = plot_residuals(f,
                             plot_unweighted=False,
                             is_right_side=True,
                             num_samples=0)
    assert issubclass(type(fig),matplotlib.figure.Figure)
    assert issubclass(type(ax),matplotlib.axes.Axes)
    plt.close(fig)

    fig, ax = plot_residuals(f,
                             plot_unweighted=False,
                             is_right_side=False,
                             num_samples=10)
    assert issubclass(type(fig),matplotlib.figure.Figure)
    assert issubclass(type(ax),matplotlib.axes.Axes)
    plt.close(fig)

    fig, ax = plot_residuals(f,
                             plot_unweighted=False,
                             is_right_side=False,
                             num_samples=0)
    assert issubclass(type(fig),matplotlib.figure.Figure)
    assert issubclass(type(ax),matplotlib.axes.Axes)
    plt.close(fig)

    fig, ax = plt.subplots(1,figsize=(6,6))
    fig_out, ax_out = plot_residuals(f,ax=ax)
    assert fig_out is fig
    assert ax_out is ax
    plt.close(fig)

    with pytest.raises(ValueError):
        plot_residuals(f,ax="not_an_ax")
    



    