
import pytest

from dataprob.fitters.ml import MLFitter
from dataprob.plot.plot_corner import plot_corner

import numpy as np
import pandas as pd
import matplotlib


def test_plot_corner():
    
    # tests run the whole decision tree of the function to identify major 
    # errors, but I'm not checking output here because it's graphical. 
    
    # some test data
    y_obs = np.arange(10)
    y_std = np.ones(10)
    def test_fcn(a=1,b=2): return a*b*np.ones(10)
    fake_samples = np.random.normal(loc=0,scale=1,size=(1000,2))

    # Create a fitter that has apparently been run and has some samples
    f = MLFitter(some_function=test_fcn)
    f.y_obs = y_obs
    f.y_std = y_std
    f._fit_df = pd.DataFrame({"name":["a","b"],"estimate":[10,20]})
    f._success = False
    f._samples = fake_samples

    # no fit_type specified
    with pytest.raises(RuntimeError):
        plot_corner(f=f)

    # set success is True, should now run
    f._success = True
    fig = plot_corner(f)
    assert issubclass(type(fig),matplotlib.figure.Figure)

    # Send in filter parameter possibilities. It should gracefully handle all
    # of these cases. 
    fig = plot_corner(f,filter_params=None)
    assert issubclass(type(fig),matplotlib.figure.Figure)
    
    fig = plot_corner(f,filter_params="blah")
    assert issubclass(type(fig),matplotlib.figure.Figure)

    fig = plot_corner(f,filter_params=["blah"])
    assert issubclass(type(fig),matplotlib.figure.Figure)

    fig = plot_corner(f,filter_params=[1])
    assert issubclass(type(fig),matplotlib.figure.Figure)
    
    # filter all parameters
    with pytest.raises(ValueError):
        fig = plot_corner(f,filter_params=["a","b"])
    
    # filter one
    fig = plot_corner(f,filter_params=["a"])
    assert issubclass(type(fig),matplotlib.figure.Figure)

    # filter other
    fig = plot_corner(f,filter_params=["b"])
    assert issubclass(type(fig),matplotlib.figure.Figure)

    # Get rid of samples attribute. Should now fail 
    f._samples = None
    with pytest.raises(RuntimeError):
        fig = plot_corner(f,)

    # put samples back in
    f._samples = fake_samples
    fig = plot_corner(f,filter_params=None)
    assert issubclass(type(fig),matplotlib.figure.Figure)

    # pass in labels
    fig = plot_corner(f,filter_params=None,labels=["x","y"])
    assert issubclass(type(fig),matplotlib.figure.Figure)

    # pass in range
    fig = plot_corner(f,filter_params=None,range=[(-10,10),(-100,100)])
    assert issubclass(type(fig),matplotlib.figure.Figure)

    # pass in truths
    fig = plot_corner(f,filter_params=None,truths=[1,2])
    assert issubclass(type(fig),matplotlib.figure.Figure)

