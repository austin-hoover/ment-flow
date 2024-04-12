from typing import Callable
from typing import List

import numpy as np
import proplot as pplt
import torch

import mentflow as mf
from mentflow.utils import coords_from_edges
from mentflow.utils import grab
from mentflow.utils import unravel


def set_proplot_rc() -> None:
    pplt.rc["cmap.discrete"] = False
    pplt.rc["cmap.sequential"] = pplt.Colormap("dark_r", space="hpl")
    pplt.rc["cycle"] = "538"
    pplt.rc["grid"] = False
    pplt.rc["figure.facecolor"] = "white"


def plot_image(image, coords=None, edges=None, ax=None, **kws):
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

    if edges is None:
        if coords is not None:
            edges = pplt.edges(coords)
        else:
            edges = [np.arange(s) for s in image.shape]
            
    return ax.pcolormesh(edges[0], edges[1], image.T, **kws)


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


def plot_proj_1d(y_meas, y_pred, edges, maxcols=7, height=1.3, colors=None, ymax=1.25, **kws):
    """Plots measured vs. predicted one-dimensional profiles."""
    if colors is None:
        colors = ["red4", "black"]

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


def plot_proj_2d(y_meas, y_pred, edges, maxcols=8, ymax=1.25, fig_kws=None, **kws):
    """Plots measured vs. predicted two-dimensional profiles."""
    if fig_kws is None:
        fig_kws = dict()
    fig_kws.setdefault("xticks", [])
    fig_kws.setdefault("yticks", [])
    fig_kws.setdefault("xspineloc", "neither")
    fig_kws.setdefault("yspineloc", "neither")
    fig_kws.setdefault("space", 0.0)

    ncols = min(len(y_meas), maxcols)
    nrows = 2 * int(np.ceil(len(y_meas) / ncols))
    figwidth = min(1.75 * ncols, 10.0)
    fig, axs = pplt.subplots(ncols=ncols, nrows=nrows, figwidth=figwidth, **fig_kws)

    i = 0
    for row in range(0, nrows, 2):
        for col in range(ncols):
            if i < len(y_meas):
                ax_index = row * ncols + col
                scale = np.max(y_meas[i])
                plot_image(y_meas[i] / scale, edges[i], ax=axs[ax_index], **kws)
                plot_image(y_pred[i] / scale, edges[i], ax=axs[ax_index + ncols], **kws)
            i += 1
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


def plot_dist_radial_pdf(
    x1, x2, rmax=3.5, bins=50, fig_kws=None, colors=None, ymax=1.25, normalize=True, **kws
):
    if colors is None:
        colors = ["red4", "black"]

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
            hist_r1[i] = hist_r1[i] / mf.utils.sphere_shell_volume(
                rmin=rmin, rmax=rmax, d=x1.shape[1]
            )
            hist_r2[i] = hist_r2[i] / mf.utils.sphere_shell_volume(
                rmin=rmin, rmax=rmax, d=x2.shape[1]
            )

    fig, ax = pplt.subplots(**fig_kws)
    scale = hist_r1.max()
    plot_profile(hist_r1 / scale, bin_edges, ax=ax, color=colors[0], **kws)
    plot_profile(hist_r2 / scale, bin_edges, ax=ax, color=colors[1], **kws)
    ax.format(ymax=ymax)
    return fig, ax


def plot_dist_radial_cdf(
    x1,
    x2,
    rmax=3.5,
    bins=50,
    fig_kws=None,
    colors=None,
    ymax=1.20,
    plot_gaussian_and_kv=False,
    **kws
):
    if colors is None:
        colors = ["red4", "black"]

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


def plot_dist_corner(x1, x2, cmaps=None, colors=None, **kws):
    import psdist.visualization as psv

    if cmaps is None:
        cmaps = [
            pplt.Colormap("blues"),
            pplt.Colormap("reds"),
        ]
    if colors is None:
        colors = ["blue6", "red6"]
    diag_kws = kws.pop("diag_kws", {})
    diag_kws.setdefault("lw", 1.5)

    grid = psv.CornerGrid(d=x1.shape[1], corner=False)
    grid.plot_points(
        x1, upper=False, cmap=cmaps[0], diag_kws=dict(color=colors[0], **diag_kws), **kws
    )
    grid.plot_points(
        x2, lower=False, cmap=cmaps[1], diag_kws=dict(color=colors[1], **diag_kws), **kws
    )
    return (grid.fig, grid.axs)


class PlotProj1D:
    def __init__(self, **kws):
        self.kws = kws

    def __call__(self, y_meas=None, y_pred=None, edges=None):
        return plot_proj_1d(y_meas, y_pred, edges, **self.kws)


class PlotProj2D:
    def __init__(self, **kws):
        self.kws = kws

    def __call__(self, y_meas=None, y_pred=None, edges=None):
        return plot_proj_2d(y_meas, y_pred, edges, **self.kws)


class PlotDist2D:
    def __init__(self, **kws):
        self.kws = kws

    def __call__(self, x1, x2):
        return plot_dist_2d(x1, x2, **self.kws)


class PlotDistRadialPDF:
    def __init__(self, **kws):
        self.kws = kws

    def __call__(self, x1, x2):
        fig, ax = plot_dist_radial_pdf(x1, x2, **self.kws)
        ax.format(xlabel="Radius", ylabel="Normalized density")
        return fig, ax


class PlotDistRadialCDF:
    def __init__(self, **kws):
        self.kws = kws

    def __call__(self, x1, x2):
        fig, ax = plot_dist_radial_cdf(x1, x2, **self.kws)
        ax.format(xlabel="Radius", ylabel="CDF")
        return fig, ax


class PlotDistCorner:
    def __init__(self, **kws):
        self.kws = kws

    def __call__(self, x2, x1):
        return plot_dist_corner(x1, x2, **self.kws)


class PlotModel:
    """Visualize predicted distribution and projections."""

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
                self.plot_proj = [
                    self.plot_proj,
                ]
        if self.plot_dist is not None:
            if type(self.plot_dist) not in [list, tuple]:
                self.plot_dist = [
                    self.plot_dist,
                ]

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
        predictions = mf.simulate.forward(x_pred, model.transforms, model.diagnostics)

        figs = []

        ## Plot model vs. true samples.
        if self.plot_dist is not None:
            for function in self.plot_dist:
                fig, axs = function(grab(x_true), grab(x_pred))
                figs.append(fig)

        ## Plot measured vs. simulated projections.
        if self.plot_proj is not None:
            y_meas = [grab(measurement) for measurement in unravel(model.measurements)]
            y_pred = [grab(prediction) for prediction in unravel(predictions)]
            edges = []
            for index, transform in enumerate(model.transforms):
                for diagnostic in model.diagnostics[index]:
                    if type(diagnostic.edges) in [tuple, list]:
                        edges.append([grab(e) for e in diagnostic.edges])
                    else:
                        edges.append(grab(diagnostic.edges))
            for function in self.plot_proj:
                fig, axs = function(y_meas, y_pred, edges)
                figs.append(fig)

        return figs
