import numpy as np
import torch


def coords_from_edges(edges):
    """Compute bin centers from evenly spaced bin edges."""
    return 0.5 * (edges[:-1] + edges[1:])


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
