"""
Code to create a "corner plot" that shows distributions and correlations of 
values for all fit parameters. 
"""

import corner
import numpy as np

import re
import warnings

def plot_corner(f,filter_params=None,**kwargs):
    """
    Create a "corner plot" that shows distributions and correlations of 
    values for all fit parameters. This can only be run if the analysis 
    has generated samples. 

    Parameters
    ----------
    f : dataprob.Fitter
        dataprob.Fitter instance for which .fit() has been run
    filter_params : list-like, optional
        strings used to search parameter names.  If a parameter name matches
        one of the patterns, it is *not* plotted. 
    **kwargs : 
        any extra keyword arguments are passed directly to corner.corner 
        to tune formatting, etc. To learn more, ``import corner`` then
        ``help(corner.corner)``. 

    Returns
    -------
    fig : matplotlib.Figure
        matplotlib figure instance generated by calling corner.corner
    """

    # Make sure fit was successful before trying to plot anything
    if not f.success:
        err = "fit has not been successfully run. cannot generate plot.\n"
        raise RuntimeError(err)
    
    # if filter parameters are not specified, no skip_pattern
    if filter_params is None:
        skip_pattern = None

    # otherwise
    else:

        # If the user passes a string (instead of a list or tuple of patterns),
        # convert it to a list up front.
        if type(filter_params) is str:
            filter_params = (filter_params,)

        # Make sure it's strings
        filter_params = [str(p) for p in filter_params]

        # compile a pattern to look for
        skip_pattern = re.compile("|".join(filter_params))

    # Check for samples
    if f.samples is None:
        w = "Fit does not have samples. Could not generate a corner plot.\n"
        warnings.warn(w)
        return None
    
    # Go through samples
    keep_indexes = []
    corner_range = []
    names = []
    est_values = []
    for i in range(f.samples.shape[1]):

        idx = f.fit_df.index[i]
        
        # Get name and estimate
        name = f.fit_df.loc[idx,"name"]
        estimate = f.fit_df.loc[idx,"estimate"]

        # look for patterns to skip
        if skip_pattern is not None and skip_pattern.search(name):
            print("not doing corner plot for parameter ",name)
            continue

        names.append(name)
        keep_indexes.append(i)

        # use nanmin in case there is a failed sample in there somewhere
        corner_range.append(tuple([np.nanmin(f.samples[:,i])-0.5,
                                   np.nanmax(f.samples[:,i])+0.5]))
        est_values.append(estimate)

    # make sure we kept at least one parameter
    if len(keep_indexes) == 0:
        err = "filter_params removed all parameters. Could not generate\n"
        err += "corner plot\n"
        raise ValueError(err)

    # Create array to plot samples
    to_plot = f.samples[:,np.array(keep_indexes,dtype=int)]

    # Load labels, range, and truths into kwargs only if the user has not
    # defined them as explicit kwargs. User corner.corner to check sanity 
    # of their inputs. 
    if "labels" not in kwargs:
        kwargs["labels"] = names
    if "range" not in kwargs:
        kwargs["range"] = corner_range
    if "truths" not in kwargs:
        kwargs["truths"] = est_values

    # Call corner 
    fig = corner.corner(to_plot,**kwargs)

    return fig