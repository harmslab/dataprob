
import pytest

from dataprob.fitters.ml import MLFitter

from dataprob.plot.plot_fit import plot_fit

import numpy as np
import matplotlib
from matplotlib import pyplot as plt

def test_plot_fit():

    # covers whole decision tree, but not an amazing tests of outputs. 
    # graphical. 

    def test_fcn(m,b): return m*np.arange(10) + b
    y_obs = test_fcn(m=5,b=1)
    y_std = np.ones(10)*0.1

    f = MLFitter(some_function=test_fcn)

    # run on fitter prior to fit. Should fail gracefully
    fig, ax = plot_fit(f)
    assert issubclass(type(fig),matplotlib.figure.Figure)
    assert issubclass(type(ax),matplotlib.axes.Axes)
    plt.close(fig)
    
    f = MLFitter(some_function=test_fcn)
    f.fit(y_obs=y_obs,
          y_std=y_std)

    fig, ax = plot_fit(f)
    assert issubclass(type(fig),matplotlib.figure.Figure)
    assert issubclass(type(ax),matplotlib.axes.Axes)
    plt.close(fig)

    with pytest.raises(ValueError):
        plot_fit(f=f,
                 num_samples=-1)
        
    fig, ax = plot_fit(f,num_samples=0)
    assert issubclass(type(fig),matplotlib.figure.Figure)
    assert issubclass(type(ax),matplotlib.axes.Axes)
    plt.close(fig)

    fig, ax = plot_fit(f,num_samples=10)
    assert issubclass(type(fig),matplotlib.figure.Figure)
    assert issubclass(type(ax),matplotlib.axes.Axes)
    plt.close(fig)

    fig, ax = plt.subplots(1,figsize=(6,6))
    fig_out, ax_out = plot_fit(f,num_samples=10,ax=ax)
    assert fig_out is fig
    assert ax_out is ax
    plt.close(fig)

    with pytest.raises(ValueError):
        plot_fit(f,ax="not_an_ax")
    