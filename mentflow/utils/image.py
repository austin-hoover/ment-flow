import numpy as np
import skimage.transform


def centers_from_edges(edges):
    """Compute bin centers from evenly spaced bin edges."""
    return 0.5 * (edges[:-1] + edges[1:])


def edges_from_centers(centers):
    """Compute bin edges from evenly spaced bin centers."""
    delta = np.diff(centers)[0]
    return np.hstack([centers - 0.5 * delta, [centers[-1] + 0.5 * delta]])


def sample_hist(hist, coords=None, n=1):
    """Sample from histogram.

    Parameters
    ----------
    hist : ndarray
        A d-dimensional histogram.
    coords : list[ndarray], length d
        Coordinates (bin centers) along each axis.
    n : int
        The number of samples to draw.

    Returns
    -------
    ndarray, shape (samples, d)
        Samples drawn from the distribution.
    """
    d = hist.ndim

    if coords is None:
        coords = [np.arange(hist.shape[k]) for k in range(d)]
    elif d == 1:
        coords = [coords]
    edges = [edges_from_centers(c) for c in coords]
    
    idx = np.flatnonzero(hist)
    pdf = hist.ravel()[idx]
    pdf = pdf / np.sum(pdf)
    idx = np.random.choice(idx, n, replace=True, p=pdf)
    idx = np.unravel_index(idx, shape=hist.shape)
    lb = [edges[axis][idx[axis]] for axis in range(d)]
    ub = [edges[axis][idx[axis] + 1] for axis in range(d)]
    return np.squeeze(np.random.uniform(lb, ub).T)


def get_grid_points(coords):
    """Return list of grid coordinates from coordinate arrays along each axis.

    Parameters
    ----------
    coords : list[ndarray]
        Coordinates along each axis of regular grid. Example: [[1, 2, 3], [0, 1, 2]].

    Returns
    -------
    ndarray, shape (K, len(coords))
        Coordinate array for all points in the grid. The total number of grid
        points is `K = np.prod([len(c) for c in coords])`.
    """
    return np.stack([C.ravel() for C in np.meshgrid(*coords, indexing="ij")], axis=-1)


def set_image_shape(image, coords, shape):
    image = skimage.transform.resize(image, shape)
    coords = [np.linspace(coords[i][0], coords[i][-1], shape[i]) for i in range(len(coords))]
    return image, coords