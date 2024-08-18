import pytest

from dataprob.plot.plot_summary import plot_summary
from dataprob.fitters.ml import MLFitter

import numpy as np
import matplotlib

def test_plot_summary():

    # There is only one logic flow in this function. The biggest thing to test
    # is whether all of the arguments are being passed correctly (without 
    # say, flipping a y_calc and a y_obs). The problem is that the outputs are
    # graphical and difficult to test directly. Leaving this under-tested for 
    # now and will rely on graphical noodling to make sure things look right, 
    # I guess...

    # Generate results with fit
    def test_fcn(m,b): return m*np.arange(10) + b
    y_obs = test_fcn(m=5,b=1)
    y_std = np.ones(10)*0.1
    f = MLFitter(some_function=test_fcn)
    f.fit(y_obs=y_obs,
          y_std=y_std)
    
    # generate summary plot
    fig = plot_summary(f=f)
    assert issubclass(type(fig),matplotlib.figure.Figure)
