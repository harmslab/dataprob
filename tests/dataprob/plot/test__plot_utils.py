
import pytest

from dataprob.fitters.ml import MLFitter

from dataprob.plot._plot_utils import get_plot_features
from dataprob.plot._plot_utils import get_style
from dataprob.plot._plot_utils import get_vectors
from dataprob.plot._plot_utils import _get_edges
from dataprob.plot._plot_utils import get_plot_dimensions
from dataprob.plot._plot_utils import sync_axes

import numpy as np
from matplotlib import pyplot as plt

def test_get_plot_features():

    def test_fcn(m,b): m*np.arange(10) + b
    f = MLFitter(some_function=test_fcn)
    
    f._success = False
    f._samples = np.ones((200,3),dtype=float)

    with pytest.raises(RuntimeError):
        get_plot_features(f,
                          x_label=None,
                          y_label=None,
                          num_samples=1)
    f._success = True
    x_label, y_label, num_samples = get_plot_features(f,
                                                      x_label=None,
                                                      y_label=None,
                                                      num_samples=1)
    assert x_label is None
    assert y_label is None
    assert num_samples == 1
    
    x_label, y_label, num_samples = get_plot_features(f,
                                                      x_label=1,
                                                      y_label="test",
                                                      num_samples=100)
    assert x_label == "1"
    assert y_label == "test"
    assert num_samples == 100

    # bad num_samples value
    with pytest.raises(ValueError):
        get_plot_features(f,
                          x_label=1,
                          y_label="test",
                          num_samples=-5)
    
    # samples are None
    f._samples = None
    with pytest.raises(ValueError):
        get_plot_features(f,
                          x_label=1,
                          y_label="test",
                          num_samples=10)
    
    # too many samples
    f._samples = np.ones((9,3),dtype=float)
    with pytest.raises(ValueError):
        get_plot_features(f,
                          x_label=1,
                          y_label="test",
                          num_samples=10)

    # should work now -- have enough samples
    get_plot_features(f,
                      x_label=1,
                      y_label="test",
                      num_samples=8)


def test_get_style():

    # Get default styles from plot.appearance
    from dataprob.plot.appearance import default_styles
    
    # Go through every default style
    for s in default_styles:

        # Make sure it's copied in properly and that a user_style of None does
        # not alter it
        out = get_style(user_style=None,
                        style_name=s)  
        assert default_styles is not out
        assert len(default_styles[s]) == len(out)
        for k in default_styles[s]:
            assert default_styles[s][k] == out[k]

        # Now make sure a correct pass updates a single field and leaves others
        # intact
        out = get_style(user_style={"some_cool_style":10},
                        style_name=s)
        assert len(default_styles[s]) + 1 == len(out)
        for k in default_styles[s]:
            assert default_styles[s][k] == out[k]
        assert out["some_cool_style"] == 10

        # bad user_style
        with pytest.raises(ValueError):
            get_style(user_style="not_a_dict",style_name=s)

    # bad style_name
    with pytest.raises(ValueError):
            get_style(user_style=None,style_name="not_a_style")

        
def test_get_vectors():

    def test_fcn(m,b): return m*np.arange(10) + b
    f = MLFitter(some_function=test_fcn)
    y_obs_in = 2*np.arange(10) + 5
    y_std_in = 0.1*np.ones(10)
    f.fit(y_obs=y_obs_in,
          y_std=y_std_in)
    
    x_axis, y_obs, y_std, y_calc = get_vectors(f=f,
                                               x_axis=None)
    assert np.array_equal(x_axis,np.arange(10))
    assert np.array_equal(y_obs,y_obs_in)
    assert np.array_equal(y_std,y_std_in)
    assert y_calc.shape == y_obs.shape

    # send in bad x_axis
    with pytest.raises(ValueError):
        get_vectors(f,x_axis=np.arange(11))
    
    # send in custom x_axis
    x_axis_in = np.arange(10,20,1)
    x_axis, y_obs, y_std, y_calc = get_vectors(f=f,
                                               x_axis=x_axis_in)
    assert np.array_equal(x_axis,x_axis_in)
    assert np.array_equal(y_obs,y_obs_in)
    assert np.array_equal(y_std,y_std_in)
    assert y_calc.shape == y_obs.shape

def test__get_edges():

    left, right = _get_edges([-3,-1],scalar=0.1)
    assert np.isclose(left,-3.2)
    assert np.isclose(right,-0.8)

    left, right = _get_edges([-3,0],scalar=0.1)
    assert np.isclose(left,-3.3)
    assert np.isclose(right,0.3)

    left, right = _get_edges([-1,1],scalar=0.1)
    assert np.isclose(left,-1.2)
    assert np.isclose(right,1.2)

    left, right = _get_edges([0,1],scalar=0.1)
    assert np.isclose(left,-0.1)
    assert np.isclose(right,1.1)

    left, right = _get_edges([1,3],scalar=0.1)
    assert np.isclose(left,0.8)
    assert np.isclose(right,3.2)

    with pytest.raises(ValueError):
        _get_edges([1,1],scalar=0.1)

def test_get_plot_dimensions():

    # main point of test is making sure values are passed in correct order
    x_values = np.arange(-10,11)
    y_values = np.arange(-5,6)

    left, right, bottom, top = get_plot_dimensions(x_values=x_values,
                                                   y_values=y_values,
                                                   scalar=0.1)
    assert left == -12
    assert right == 12
    assert bottom == -6
    assert top == 6

    left, right, bottom, top = get_plot_dimensions(x_values=x_values,
                                                   y_values=y_values,
                                                   scalar=0.2)
    assert left == -14
    assert right == 14
    assert bottom == -7
    assert top == 7

def test_sync_axes():

    _, ax1 = plt.subplots(1,figsize=(6,6))
    _, ax2 = plt.subplots(1,figsize=(6,6))

    ax1.set_xlim((-1,1))
    ax1.set_ylim((-1,1))
    ax2.set_xlim((0,2))
    ax2.set_ylim((0,2))

    sync_axes(ax1,ax2,'x')
    assert ax1.get_xlim() == (-1,2)
    assert ax2.get_xlim() == (-1,2)
    assert ax1.get_ylim() == (-1,1)
    assert ax2.get_ylim() == (0,2)

    ax1.set_xlim((-1,1))
    ax1.set_ylim((-1,1))
    ax2.set_xlim((0,2))
    ax2.set_ylim((0,2))

    sync_axes(ax1,ax2,'y')
    assert ax1.get_xlim() == (-1,1)
    assert ax2.get_xlim() == (0,2)
    assert ax1.get_ylim() == (-1,2)
    assert ax2.get_ylim() == (-1,2)
    

    