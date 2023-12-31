import numpy as np
import skimage.transform
import torch

from .utils import grab


def centers_from_edges(edges):
    """Compute bin centers from evenly spaced bin edges."""
    return 0.5 * (edges[:-1] + edges[1:])


def sample_hist(hist, bin_edges=None, n=1):
    """Sample from histogram.

    Parameters
    ----------
    hist : ndarray
        A d-dimensional histogram.
    bin_edges : list[ndarray], shape (d + 1,)
        Coordinates of bin edges along each axis.
    n : int
        The number of samples to draw.

    Returns
    -------
    ndarray, shape (n, d)
        Samples drawn from the distribution.
    """
    d = hist.ndim

    if bin_edges is None:
        bin_edges = [np.arange(s + 1) for s in hist.shape]
    elif d == 1:
        bin_edges = [bin_edges]
    
    idx = np.flatnonzero(hist)
    pdf = hist.ravel()[idx]
    pdf = pdf / np.sum(pdf)
    idx = np.random.choice(idx, n, replace=True, p=pdf)
    idx = np.unravel_index(idx, shape=hist.shape)
    lb = [bin_edges[axis][idx[axis]] for axis in range(d)]
    ub = [bin_edges[axis][idx[axis] + 1] for axis in range(d)]
    return np.squeeze(np.random.uniform(lb, ub).T)


def sample_hist_torch(hist, bin_edges=None, n=1):
    _hist = grab(hist)
    _bin_edges = [grab(e) for e in bin_edges]
    x = sample_hist(_hist, _bin_edges, n)
    x = torch.from_numpy(x)
    return x


def get_grid_points(*coords):
    """Return list of grid coordinates from coordinate arrays along each axis.

    Parameters
    ----------
    coords : iterable[ndarray]
        Coordinates along each axis of regular grid. Example: [[1, 2, 3], [0, 1, 2]].

    Returns
    -------
    ndarray, shape (K, len(coords))
        Coordinate array for all points in the grid. The total number of grid
        points is `K = np.prod([len(c) for c in coords])`.
    """
    return np.vstack([C.ravel() for C in np.meshgrid(*coords, indexing="ij")]).T


def get_grid_points_torch(*coords):
    return torch.vstack([C.ravel() for C in torch.meshgrid(*coords, indexing="ij")]).T


def set_image_shape(image, edges, shape):
    image = skimage.transform.resize(image, shape)
    edges = [np.linspace(edges[i][0], edges[i][-1], shape[i] + 1) for i in range(len(edges))]
    return image, edges