import numpy as np
import matplotlib.pyplot as plt
import proplot as pplt

from experiments.plotting import plot_image
from experiments.plotting import plot_hist
from experiments.plotting import plot_points
from experiments.plotting import plot_proj_1d as plot_proj


def plot_dist(x0, x, prob=None, coords=None, bins=75, limits=None, **kws):
    ncols = 3
    if prob is None:
        ncols = 2
    fig, axs = pplt.subplots(ncols=ncols, xspineloc="neither", yspineloc="neither", space=0.0, share=False)

    plot_points(x0, bins=bins, limits=limits, ax=axs[0], **kws)
    plot_points(x, bins=bins, limits=limits, ax=axs[1], **kws)
    if ncols == 3:
        plot_image(prob, coords=coords, ax=axs[2], **kws)                
    axs.format(xlim=limits[0], ylim=limits[1])
    return fig, axs
