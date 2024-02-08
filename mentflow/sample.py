from typing import Callable
from typing import List
from typing import Tuple

import torch


def sample_hist_bins(hist: torch.Tensor, n: int) -> torch.Tensor:
    pdf = torch.ravel(hist)
    idx = torch.squeeze(torch.nonzero(pdf))
    pdf = pdf[idx]
    pdf = pdf / torch.sum(pdf)
    idx = idx[pdf.multinomial(num_samples=n, replacement=True)]
    idx = torch.unravel_index(idx, hist.shape)
    return idx


def sample_hist(hist: torch.Tensor, bin_edges: List[torch.Tensor] = None, n: int = 1, noise: float = 0.0):
    d = hist.ndim

    if bin_edges is None:
        bin_edges = [torch.arange(hist.shape[axis] + 1) for axis in range(d)]
    
    if d == 1 and len(bin_edges) > 1:
        bin_edges = [bin_edges]

    idx = sample_hist_bins(hist, n)
    
    x = torch.zeros(n, d)
    for axis in range(d):
        lb = bin_edges[axis][idx[axis]]
        ub = bin_edges[axis][idx[axis] + 1]
        x[:, axis] = lb + (ub - lb) * torch.rand(n)
        if noise:
            x[:, axis] += noise * (ub - lb) * (torch.rand(n) - 0.5)
    x = torch.squeeze(x)
    return x


class GridSampler:
    def __init__(self, limits=None, shape=None, noise=0.0, device=None, store=True):
        self.device = device
        self.shape = shape
        self.limits = limits
        self.d = len(limits)
        self.noise = noise
        self.grid_edges = [
            torch.linspace(self.limits[axis][0], self.limits[axis][1], self.shape[axis] + 1)
            for axis in range(self.d)
        ]
        self.grid_coords = [0.5 * (e[:-1] + e[1:]) for e in self.grid_edges]

        self.store = store
        self.grid_points = None

    def send(self, x):
        return x.type(torch.float32).to(self.device)

    def get_mesh(self) -> Tuple[torch.Tensor]:
        return torch.meshgrid(*self.grid_coords, indexing="ij")

    def get_grid_points(self) -> torch.Tensor:
        if self.grid_points is not None:
            return self.grid_points
        grid_points = torch.vstack([C.ravel() for C in self.get_mesh()]).T
        grid_points = self.send(grid_points)
        if self.store:
            self.grid_points = grid_points
        return grid_points

    def __call__(self, log_prob_func: Callable, n: int) -> torch.Tensor:
        values = torch.exp(log_prob_func(self.get_grid_points()))
        values = torch.reshape(values, self.shape)
        x = sample_hist(values, bin_edges=self.grid_edges, n=n, noise=self.noise)
        x = self.send(x)
        return x

    def to(self, device):
        self.device = device
        self.grid_edges = [self.send(e) for e in self.grid_edges]
        self.grid_coords = [self.send(c) for c in self.grid_coords]
        return self
