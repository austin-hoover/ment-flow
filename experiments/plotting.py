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