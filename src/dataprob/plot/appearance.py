"""
Styling information for dataprob plots.
"""

default_styles = {

    "y_calc":{
        "linestyle":"-",
        "color":"red",
        "lw":2,
        "zorder":10
    },
    "y_obs":{
        "marker":"o",
        "markeredgecolor":"black",
        "markerfacecolor":"none",
        "markersize":4,
        "lw":0,
        "zorder":5
    },
    "y_std":{
        "lw":0,
        "ecolor":"black",
        "elinewidth":1,
        "capsize":3,
        "zorder":4
    },
    "sample_line":{
        "linestyle":"-",
        "alpha":0.1,
        "color":"black",
        "zorder":0
    },
    "sample_point":{
        "marker":"o",
        "alpha":0.1,
        "markersize":4,
        "markeredgecolor":"black",
        "markerfacecolor":"gray",
        "linewidth":0,
        "zorder":0
    },
    "hist_bar":{
        "lw":1,
        "edgecolor":"black",
        "facecolor":"lightgray",
        "zorder":5
    }

}
