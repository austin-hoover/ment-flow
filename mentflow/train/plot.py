from typing import Callable
import numpy as np
import proplot as pplt
import torch

import mentflow as mf
from mentflow.utils import grab
from mentflow.utils import unravel


def set_proplot_rc():
    """Set proplot style."""
    pplt.rc["cmap.discrete"] = False
    pplt.rc["cmap.sequential"] = pplt.Colormap("dark_r", space="hpl")
    pplt.rc["cycle"] = "538"
    pplt.rc["grid"] = False


def plot_image(image, coords=None, ax=None, **kws):
    """Plot two-dimensional image."""
    
    kws.setdefault("ec", "None")
    kws.setdefault("linewidth", 0.0)
    kws.setdefault("rasterized", True)
    kws.setdefault("shading", "auto")
    kws.setdefault("norm", None)
    kws.setdefault("colorbar", False)
    kws.setdefault("colorbar_kw", dict())

    image = image.copy()

    if kws["norm"] == "log":
        kws["colorbar_kw"].setdefault("formatter", "log")
        if np.count_nonzero(image):
            image + np.min(image[image > 0])
        image = np.ma.masked_less_equal(image, 0.0)
    return ax.pcolormesh(coords[0], coords[1], image.T, **kws)


def plot_points(x, bins=None, limits=None, ax=None, **kws):
    """Compute and plot two-dimensional histogram."""
    hist, edges = np.histogramdd(x, bins=bins, range=limits, density=True)
    coords = [0.5 * (e[1:] + e[:-1]) for e in edges]
    return plot_image(hist, coords, ax=ax, **kws)


def plot_profile(heights, edges, ax=None, kind="step", **kws):
    """Plot profile using line or stairs."""
    kws.setdefault("lw", 1.25)
    if kind == "step":
        return ax.stairs(heights, edges, **kws)
    else:
        coords = 0.5 * (edges[:-1] + edges[1:])
        return ax.plot(coords, heights, **kws)


def plot_proj_1d(
    y_meas,
    y_pred,
    edges,
    maxcols=7,
    height=1.3,
    colors=None,
    ymax=1.25,
    **kws
):
    """Plots measured vs. predicted one-dimensional profiles."""
    if colors is None:
        colors = ["black", "red"]
        
    ncols = min(len(y_meas), maxcols)
    nrows = int(np.ceil(len(y_meas) / ncols))
    figheight = height * nrows
    figwidth = 1.75 * ncols
    fig, axs = pplt.subplots(ncols=ncols, nrows=nrows, figheight=figheight, figwidth=figwidth)
    for j, ax in enumerate(axs):
        scale = np.max(y_meas[j])
        plot_profile(y_meas[j] / scale, edges[j], ax=ax, color=colors[0], **kws)
        if y_pred is not None:
            plot_profile(y_pred[j] / scale, edges[j], ax=ax, color=colors[1], **kws)
    axs.format(ymax=ymax)
    return fig, axs


def plot_dist_2d(x1, x2, fig_kws=None, **kws):
    """Plots two-dimensional histograms side-by-side."""
    fig_kws.setdefault("xspineloc", "neither")
    fig_kws.setdefault("yspineloc", "neither")
    fig_kws.setdefault("space", 0.0)
    fig_kws.setdefault("share", False)

    fig, axs = pplt.subplots(ncols=2, **fig_kws)
    plot_points(x1, ax=axs[0], **kws)
    plot_points(x2, ax=axs[1], **kws)
    return fig, axs


class PlotProj1D:
    def __init__(
        self,
        maxcols=7,
        height=1.3,
        colors=None,
        ymax=1.25,
        **kws
    ):    
        """Plots measured vs. predicted one-dimensional profiles."""
        self.maxcols = maxcols
        self.height = height
        self.colors = colors
        if self.colors is None:
            self.colors = ["black", "red"]
        self.ymax = ymax
        self.kws = kws
        
    def __call__(self, y_meas=None, y_pred=None, edges=None):
        fig, axs = plot_proj_1d(
            y_meas,
            y_pred,
            edges,
            maxcols=self.maxcols,
            height=self.height,
            colors=self.colors,
            ymax=self.ymax,
            **self.kws
        )
        return fig, axs
        

class PlotDist2D:
    """Plots two-dimensional histograms side-by-side."""
    def __init__(self, fig_kws=None, **kws):
        self.fig_kws = fig_kws
        if self.fig_kws  is None:
            self.fig_kws  = dict()
        self.fig_kws.setdefault("xspineloc", "neither")
        self.fig_kws.setdefault("yspineloc", "neither")
        self.fig_kws.setdefault("space", 0.0)
        self.fig_kws.setdefault("share", False)
        self.kws = kws

    def __call__(self, x1, x2):
        fig, axs = plot_dist_2d(x1, x2, fig_kws=self.fig_kws, **self.kws)
        return fig, axs


class PlotModel:
    """Visualize predicted distribution and projections.

    This class should work for any input/output dimension.
    """
    def __init__(
        self, 
        dist: Callable,
        n_samples: int, 
        plot_proj: Callable,
        plot_dist: Callable, 
        device: str = "cpu",
    ):
        """Constructor.

        Parameters
        ----------
        dist : callable
            Implements `dist.sample(n)` to draw samples from the true distribution.
        n_samples : int
            Number of samples to plot.
        plot_dist: callable
            Plots samples from true and predicted distributions.
            Signature: `plot_dist(x_true, x_pred)`.
        plot_proj: callable
            Plots simulated vs. actual measurements.
            Signature: `plot_proj(y_meas, y_pred, edges)`.
        device : str
            Name of torch device.
        """
        self.dist = dist
        self.n_samples = n_samples
        self.plot_proj = plot_proj
        self.plot_dist = plot_dist
        self.device = torch.device(device)

    def send(self, x):
        """Send x to device."""
        x = x.type(torch.float32)
        if self.device is not None:
            x = x.to(self.device)
        return x

    def __call__(self, model):
        """Return list of figures."""
        # Generate particles.
        x_true = self.dist.sample(self.n_samples)
        x_true = self.send(x_true)
        x_pred = model.sample(x_true.shape[0])
        x_pred = self.send(x_pred)
    
        # Simulate measurements.
        predictions = mf.sim.forward(x_pred, model.transforms, model.diagnostics)

        figs = []
        
        ## Plot model vs. true samples.
        fig, axs = self.plot_dist(grab(x_true), grab(x_pred))
        figs.append(fig)
    
        ## Plot measured vs. simulated projections.
        y_meas = [grab(measurement) for measurement in unravel(model.measurements)]
        y_pred = [grab(prediction) for prediction in unravel(predictions)]
        edges = []
        for index, transform in enumerate(model.transforms):
            for diagnostic in model.diagnostics[index]:
                edges.append(grab(diagnostic.bin_edges))
                
        fig, axs = self.plot_proj(y_meas, y_pred, edges)
        figs.append(fig)
        
        return figs

