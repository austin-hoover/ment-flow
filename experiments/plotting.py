import numpy as np
import proplot as pplt


def plot_image(image, coords=None, ax=None, **kws):
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
    hist, edges = np.histogramdd(x, bins=bins, range=limits, density=True)
    coords = [0.5 * (e[1:] + e[:-1]) for e in edges]
    return plot_image(hist, coords=coords, ax=ax, **kws)


def plot_hist(heights, edges, ax=None, kind="step", **kws):
    kws.setdefault("lw", 1.25)
    if kind == "step":
        return ax.stairs(heights, edges, **kws)
    else:
        coords = 0.5 * (edges[:-1] + edges[1:])
        return ax.plot(coords, heights, **kws)


def plot_proj_1d(
    measurements,
    predictions=None,
    bin_edges=None,
    maxcols=7,
    kind="step",
    height=1.3,
    colors=None,
    ymax=1.25,
    **kws
):
    if colors is None:
        colors = ["black", "red"]
    ncols = min(len(measurements), maxcols)
    nrows = int(np.ceil(len(measurements) / ncols))
    figheight = height * nrows
    figwidth = 1.75 * ncols
    fig, axs = pplt.subplots(ncols=ncols, nrows=nrows, figheight=figheight, figwidth=figwidth)
    for j, (measurement, ax) in enumerate(zip(measurements, axs)):
        scale = np.max(measurement)
        plot_hist(measurements[j] / scale, bin_edges, ax=ax, kind=kind, color=colors[0], **kws)
        if predictions is not None:
            plot_hist(predictions[j] / scale, bin_edges, ax=ax, kind=kind, color=colors[1], **kws)
    axs.format(ymax=ymax)
    return fig, axs
