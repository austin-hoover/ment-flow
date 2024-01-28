import math
from typing import Callable
from typing import Iterable
from typing import List
from typing import Tuple

import numpy as np
import torch

from mentflow.utils import grab


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


class GridSampler:
    def __init__(self, limits=None, res=50, noise=0.0, device=None):
        self.device = device
        self.res = res
        self.limits = limits
        self.d = len(limits)
        self.noise = noise
        self.initialize(limits=limits, res=res)
        self.to(device)

    def initialize(self, limits=None, res=None):
        if limits is not None:
            self.limits = limits
        if res is not None:
            self.res = res
        self.d = len(limits)
        self.grid_edges = [
            torch.linspace(self.limits[i][0], self.limits[i][1], self.res + 1) 
            for i in range(self.d)
        ]
        self.grid_coords = [0.5 * (e[:-1] + e[1:]) for e in self.grid_edges]
        self.shape = self.d * [self.res]

    def send(self, x):
        return x.type(torch.float32).to(self.device)

    def to(self, device):
        self.device = device
        self.grid_edges = [self.send(e) for e in self.grid_edges]
        self.grid_coords = [self.send(c) for c in self.grid_coords]
        return self

    def get_mesh(self) -> List[torch.Tensor]:
        return torch.meshgrid(*self.grid_coords, indexing="ij")

    def get_grid_points(self) -> torch.Tensor:
        return torch.vstack([C.ravel() for C in self.get_mesh()]).T
        
    def __call__(self, log_prob_func: Callable, n: int) -> torch.Tensor:
        grid_points = self.get_grid_points()
        grid_points = self.send(grid_points)
        log_values = log_prob_func(grid_points)
        values = torch.exp(log_values)
        values = values.reshape(self.shape)
        x = sample_hist_torch(values, bin_edges=self.grid_edges, n=n)
        x = self.send(x)
        if self.noise:
            for j in range(x.shape[1]):
                delta = self.noise * (self.grid_edges[j][1] - self.grid_edges[j][0])
                noise = torch.rand(x.shape[0], device=self.device) - 0.5
                noise = noise * delta
                x[:, j] = x[:, j] + noise
        return x
