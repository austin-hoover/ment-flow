import numpy as np
import matplotlib.pyplot as plt
import proplot as pplt
import torch

from experiments.plotting import plot_image
from experiments.plotting import plot_points


def plot_dist(x0, x, prob=None, coords=None, n_bins=75, limits=None, **kws):
    ncols = 3
    if prob is None:
        ncols = 2
    fig, axs = pplt.subplots(ncols=ncols, xspineloc="neither", yspineloc="neither", space=0.0, share=False)
    plot_points(x0, bins=n_bins, limits=limits, ax=axs[0], **kws)
    plot_points(x, bins=n_bins, limits=limits, ax=axs[1], **kws)
    if ncols == 3:
        plot_image(prob, coords=coords, ax=axs[2], **kws)                
    axs.format(xlim=limits[0], ylim=limits[1])
    return fig, axs
    

def plot_proj(measurements, predictions=None, bin_edges=None, maxcols=7, kind="step", height=1.3):
    
    def _plot(heights, edges, ax=None, kind="step", **kws):
        if kind == "step":
            return ax.stairs(heights, edges, **kws)
        else:
            coords = 0.5 * (edges[:-1] + edges[1:])
            return ax.plot(coords, heights, **kws)
    
    ncols = min(len(measurements), maxcols)
    nrows = int(np.ceil(len(measurements) / ncols))
    figheight = height * nrows
    figwidth = 1.75 * ncols
    fig, axs = pplt.subplots(ncols=ncols, nrows=nrows, figheight=figheight, figwidth=figwidth)
    for j in range(len(measurements)):
        ax = axs[j]
        measurement = measurements[j]
        scale = np.max(measurement)
        kws = dict(lw=1.25)
        _plot(measurement / scale, bin_edges, ax=ax, kind=kind, color="black", **kws)
        if predictions is not None:
            prediction = predictions[j]
            _plot(prediction / scale, bin_edges, ax=ax, kind=kind, color="red", **kws)
    axs.format(ymax=1.25)
    return fig, axs

