from typing import Callable
from typing import List

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
    pplt.rc["figure.facecolor"] = "white"


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

    for index in range(len(y_meas)):
        ax = axs[index]
        scale = np.max(y_meas[index])
        plot_profile(y_meas[index] / scale, edges[index], ax=ax, color=colors[0], **kws)
        if y_pred is not None:
            plot_profile(y_pred[index] / scale, edges[index], ax=ax, color=colors[1], **kws)
    axs.format(ymax=ymax)
    return fig, axs


def plot_dist_2d(x1, x2, fig_kws=None, **kws):
    """Plots two-dimensional histograms side-by-side."""
    if fig_kws is None:
        fig_kws = dict()
    fig_kws.setdefault("xspineloc", "neither")
    fig_kws.setdefault("yspineloc", "neither")
    fig_kws.setdefault("space", 0.0)
    fig_kws.setdefault("share", False)

    fig, axs = pplt.subplots(ncols=2, **fig_kws)
    plot_points(x1, ax=axs[0], **kws)
    plot_points(x2, ax=axs[1], **kws)
    return fig, axs


def plot_dist_radial_hist(x1, x2, rmax=3.5, bins=80, fig_kws=None, colors=None, ymax=1.25, normalize=True, **kws):
    if colors is None:
        colors = ["black", "red"]

    if fig_kws is None:
        fig_kws = dict()
    fig_kws.setdefault("figwidth", 3.0)
    fig_kws.setdefault("figheight", 2.0)

    bin_edges = np.linspace(0.0, rmax, bins + 1)

    r1 = np.linalg.norm(x1, axis=1)
    r2 = np.linalg.norm(x2, axis=1)
    hist_r1, _ = np.histogram(r1, bins=bin_edges, density=True)
    hist_r2, _ = np.histogram(r2, bins=bin_edges, density=True)

    if normalize:
        for i in range(len(bin_edges) - 1):
            rmin = bin_edges[i]
            rmax = bin_edges[i + 1]
            hist_r1[i] = hist_r1[i] / mf.utils.sphere_shell_volume(rmin=rmin, rmax=rmax, d=x1.shape[1])
            hist_r2[i] = hist_r2[i] / mf.utils.sphere_shell_volume(rmin=rmin, rmax=rmax, d=x2.shape[1])
    
    fig, ax = pplt.subplots(**fig_kws)
    scale = hist_r1.max()
    plot_profile(hist_r1 / scale, bin_edges, ax=ax, color=colors[0], **kws)
    plot_profile(hist_r2 / scale, bin_edges, ax=ax, color=colors[1], **kws)
    ax.format(ymax=ymax)
    return fig, ax


def plot_dist_radial_cdf(x1, x2, rmax=3.5, bins=80, fig_kws=None, colors=None, ymax=1.20, plot_gaussian_and_kv=False, **kws):
    if colors is None:
        colors = ["black", "red"]

    if fig_kws is None:
        fig_kws = dict()
    fig_kws.setdefault("figwidth", 3.0)
    fig_kws.setdefault("figheight", 2.0)

    bin_edges = np.linspace(0.0, rmax, bins + 1)

    r1 = np.linalg.norm(x1, axis=1)
    r2 = np.linalg.norm(x2, axis=1)
    hist_r1, _ = np.histogram(r1, bins=bin_edges, density=True)
    hist_r2, _ = np.histogram(r2, bins=bin_edges, density=True)

    for i in range(len(bin_edges) - 1):
        rmin = bin_edges[i]
        rmax = bin_edges[i + 1]
        hist_r1[i] = hist_r1[i] / mf.utils.sphere_shell_volume(rmin=rmin, rmax=rmax, d=x1.shape[1])
        hist_r2[i] = hist_r2[i] / mf.utils.sphere_shell_volume(rmin=rmin, rmax=rmax, d=x2.shape[1])

    # Compute radial cdf.
    cdf_r1 = np.cumsum(hist_r1)
    cdf_r2 = np.cumsum(hist_r2)
    cdf_r1 = cdf_r1 / cdf_r1[-1]
    cdf_r2 = cdf_r2 / cdf_r2[-1]

    fig, ax = pplt.subplots(**fig_kws)
    plot_profile(cdf_r1, bin_edges, ax=ax, color=colors[0], **kws)
    plot_profile(cdf_r2, bin_edges, ax=ax, color=colors[1], **kws)    
    ax.format(xmin=0.0, ymax=ymax, ymin=0.0)
    return fig, ax
    

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


class PlotDistRadialHist:
    def __init__(self, rmax=5.0, bins=75, fig_kws=None, **kws):
        self.kws = kws
        self.kws["rmax"] = rmax
        self.kws["bins"] = bins
        self.kws["fig_kws"] = fig_kws

    def __call__(self, x1, x2):
        fig, ax = plot_dist_radial_hist(x1, x2, **self.kws)
        ax.format(xlabel="Radius", ylabel="Normalized density")
        return fig, ax


class PlotDistRadialCDF:
    def __init__(self, rmax=5.0, bins=75, fig_kws=None, **kws):
        self.kws = kws
        self.kws["rmax"] = rmax
        self.kws["bins"] = bins
        self.kws["fig_kws"] = fig_kws

    def __call__(self, x1, x2):
        fig, ax = plot_dist_radial_cdf(x1, x2, **self.kws)
        ax.format(xlabel="Radius", ylabel="CDF")
        return fig, ax


class PlotModel:
    """Visualize predicted distribution and projections.

    This class should work for any input/output dimension.
    """
    def __init__(
        self, 
        dist: Callable,
        n_samples: int, 
        plot_proj: List[Callable],
        plot_dist: List[Callable], 
        device=None,
    ):
        """Constructor.

        Parameters
        ----------
        dist : callable
            Implements `dist.sample(n)` to draw samples from the true distribution.
        n_samples : int
            Number of samples to plot.
        plot_dist: list[callable]
            Plots samples from true and predicted distributions.
            Signature: `plot_dist(x_true, x_pred)`.
        plot_proj: list[callable]
            Plots simulated vs. actual measurements.
            Signature: `plot_proj(y_meas, y_pred, edges)`.
        device : str
            Name of torch device.
        """
        self.dist = dist
        self.n_samples = n_samples
        self.plot_proj = plot_proj
        self.plot_dist = plot_dist
        self.device = device

        if self.plot_proj is not None:
            if type(self.plot_proj) not in [list, tuple]:
                self.plot_proj = [self.plot_proj,]
        if self.plot_dist is not None:
            if type(self.plot_dist) not in [list, tuple]:
                self.plot_dist = [self.plot_dist,]

    def send(self, x):
        """Send x to device."""
        return x.type(torch.float32).to(self.device)

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
        if self.plot_dist is not None:
            for function in self.plot_dist:
                fig, axs = function(grab(x_true), grab(x_pred))
                figs.append(fig)
        
        ## Plot measured vs. simulated projections.
        y_meas = [grab(measurement) for measurement in unravel(model.measurements)]
        y_pred = [grab(prediction) for prediction in unravel(predictions)]
        edges = []
        for index, transform in enumerate(model.transforms):
            for diagnostic in model.diagnostics[index]:
                edges.append(grab(diagnostic.bin_edges))

        if self.plot_proj is not None:
            for function in self.plot_proj:
                fig, axs = function(y_meas, y_pred, edges)
                figs.append(fig)
        
        return figs
