import math
import time
from typing import Callable
from typing import List
from typing import Tuple

import numpy as np
import torch
from tqdm import tqdm

from mentflow.utils import coords_from_edges
from mentflow.utils import get_grid_points


def tqdm_wrapper(iterable, verbose=False):
    return tqdm(iterable) if verbose else iterable


def random_uniform(lb: float, ub: float, size: int, device=None) -> torch.Tensor:
    return lb + (ub - lb) * torch.rand(size, device=device)


def random_choice(items: torch.tensor, size: int, p: torch.Tensor):
    return items[p.multinomial(num_samples=size, replacement=True)]


def sample_hist_bins(hist: torch.Tensor, size: int) -> torch.Tensor:
    pdf = torch.ravel(hist) + 1.00e-15
    idx = torch.squeeze(torch.nonzero(pdf))
    idx = random_choice(idx, size, p=(pdf / torch.sum(pdf)))
    return idx


def sample_hist(
    hist: torch.Tensor,
    edges: List[torch.Tensor],
    size: int,
    noise: float = 0.0,
    device: torch.device = None,
) -> torch.Tensor:
    ndim = hist.ndim
    if ndim == 1:
        edges = [edges]

    idx = sample_hist_bins(hist, size)
    idx = torch.unravel_index(idx, hist.shape)

    x = torch.zeros((size, ndim), device=device)
    for axis in range(ndim):
        lb = edges[axis][idx[axis]]
        ub = edges[axis][idx[axis] + 1]
        x[:, axis] = random_uniform(lb, ub, size, device=device)
        if noise:
            delta = ub - lb
            x[:, axis] += 0.5 * random_uniform(-delta, delta, size, device=device)
    x = torch.squeeze(x)
    return x


class GridSampler:
    def __init__(
        self,
        limits: List[Tuple[float]],
        shape: Tuple[int],
        noise: float = 0.0,
        device: torch.device = None,
        store: bool = True,
    ) -> None:
        self.device = device
        self.shape = shape
        self.limits = limits
        self.ndim = len(limits)
        self.noise = noise
        self.store = store

        self.edges = [
            torch.linspace(
                self.limits[axis][0],
                self.limits[axis][1],
                self.shape[axis] + 1,
            )
            for axis in range(self.ndim)
        ]
        self.coords = [coords_from_edges(e) for e in self.edges]
        self.points = None

    def send(self, x: torch.Tensor) -> torch.Tensor:
        return x.type(torch.float32).to(self.device)

    def get_grid_points(self) -> torch.Tensor:
        if self.points is not None:
            return self.points
        points = get_grid_points(*self.coords)
        points = self.send(points)
        if self.store:
            self.points = points
        return points

    def __call__(self, prob_func: Callable, size: int) -> torch.Tensor:
        prob = prob_func(self.get_grid_points())
        prob = torch.reshape(prob, self.shape)
        x = sample_hist(prob, self.edges, size=size, noise=self.noise, device=self.device)
        x = self.send(x)
        return x

    def to(self, device):
        self.device = device
        self.edges = [self.send(e) for e in self.edges]
        self.coords = [self.send(c) for c in self.coords]
        return self
