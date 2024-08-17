
import pytest

from dataprob.fitters.ml import MLFitter

from dataprob.plot._plot_utils import get_plot_features
from dataprob.plot._plot_utils import validate_and_load_style
from dataprob.plot._plot_utils import get_styling
from dataprob.plot._plot_utils import get_vectors
from dataprob.plot._plot_utils import _get_edges
from dataprob.plot._plot_utils import get_plot_dimensions

import numpy as np

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


def test__validate_and_load_style():

    default_style = {"x":10}
    out = validate_and_load_style(some_style=None,
                                  some_style_name="x",
                                  default_style=default_style)
    assert len(out) == 1
    assert out["x"] == 10
    assert out is not default_style

    
    with pytest.raises(ValueError):
        validate_and_load_style(some_style="not_a_dict",
                                some_style_name="x",
                                default_style=default_style)
    
    out = validate_and_load_style(some_style={"not_x":5},
                                  some_style_name="x",
                                  default_style=default_style)
    assert len(out) == 2
    assert out["x"] == 10
    assert out["not_x"] == 5
    assert out is not default_style

    out = validate_and_load_style(some_style={"not_x":5,"x":20},
                                  some_style_name="x",
                                  default_style=default_style)
    assert len(out) == 2
    assert out["x"] == 20
    assert out["not_x"] == 5
    assert out is not default_style

def test_get_styling():

    # do not test extensively, just make sure copying and sanity checking
    # are working. this is because default styles could change and testing 
    # was done in test__validate_and_load_style

    # make sure data is being sent in, output dictionary has more than just 
    # what we sent in, and that the correct order is maintained (we didn't)
    # mix up return order or make some wacky copy-paste error in names
    y_obs_style = {"w":1}
    y_std_style = {"x":2}
    y_calc_style = {"y":3}
    sample_style = {"z":4}
    a, b, c, d = get_styling(y_obs_style=y_obs_style,
                             y_std_style=y_std_style,
                             y_calc_style=y_calc_style,
                             sample_style=sample_style)
    assert len(a) > 1
    assert a["w"] == 1

    assert len(b) > 1
    assert b["x"] == 2

    assert len(c) > 1
    assert c["y"] == 3

    assert len(d) > 1
    assert d["z"] == 4

    with pytest.raises(ValueError):
        get_styling(y_obs_style="not_a_dict",
                    y_std_style=y_std_style,
                    y_calc_style=y_calc_style,
                    sample_style=sample_style)
        
    with pytest.raises(ValueError):
        get_styling(y_obs_style=y_obs_style,
                    y_std_style="not_a_dict",
                    y_calc_style=y_calc_style,
                    sample_style=sample_style)
        
    with pytest.raises(ValueError):
        get_styling(y_obs_style=y_obs_style,
                    y_std_style=y_std_style,
                    y_calc_style="not_a_dict",
                    sample_style=sample_style)

    with pytest.raises(ValueError):
        get_styling(y_obs_style=y_obs_style,
                    y_std_style=y_std_style,
                    y_calc_style=y_calc_style,
                    sample_style="not_a_dict")
        
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
    assert not np.array_equal(y_obs,y_calc) # fit -- not exactly same

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
    assert not np.array_equal(y_obs,y_calc) # fit -- not exactly same

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

    