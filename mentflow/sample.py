import math
import time
from typing import Callable
from typing import List
from typing import Tuple

import numpy as np
import torch
from tqdm import tqdm

from mentflow.utils.grid import get_grid_points


def tqdm_wrapper(iterable, verbose=False):
    return tqdm(iterable) if verbose else iterable


def random_uniform(lb: float, ub: float, size: int) -> torch.Tensor:
    return lb + (ub - lb) * torch.rand(size)


def random_choice(items: torch.tensor, n: int, pdf: torch.Tensor):
    return items[pdf.multinomial(num_samples=n, replacement=True)]


def sample_hist_bins(hist: torch.Tensor, n: int) -> torch.Tensor:
    pdf = torch.ravel(hist)
    idx = torch.squeeze(torch.nonzero(pdf))
    pdf = pdf[idx]
    pdf = pdf / torch.sum(pdf)
    idx = random_choice(idx, n, pdf=pdf)
    idx = torch.unravel_index(idx, hist.shape)
    return idx


def sample_hist(
    hist: torch.Tensor,
    bin_edges: List[torch.Tensor] = None,
    n: int = 1,
    noise: float = 0.0,
):
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
        x[:, axis] = random_uniform(lb, ub, n)
        if noise:
            delta = ub - lb
            x[:, axis] += noise * delta * (torch.rand(n) - 0.5)
    x = torch.squeeze(x)
    return x


def resample_hist(hist, n):
    """Equivalent to sampling particles from histogram and recomputing histogram."""
    pdf = hist.ravel()
    idx = torch.squeeze(torch.nonzero(pdf))
    pdf = pdf[idx]
    pdf = pdf / torch.sum(pdf)
    idx = random_choice(idx, n, pdf=pdf)
    idx = idx.float()
    bin_edges = torch.arange(hist.numel() + 1).float() - 0.5
    hist_out = torch.histogram(idx, bins=bin_edges, density=False)
    hist_out = hist_out.hist
    hist_out = torch.reshape(hist_out, hist.shape)
    return hist_out


class GridSampler:
    def __init__(
        self,
        grid_limits: List[Tuple[float]],
        grid_shape: Tuple[int],
        noise: float = 0.0,
        device=None,
        store=True,
    ):
        self.device = device
        self.grid_shape = grid_shape
        self.grid_limits = grid_limits
        self.d = len(grid_limits)
        self.noise = noise
        self.grid_edges = [
            torch.linspace(
                self.grid_limits[axis][0],
                self.grid_limits[axis][1],
                self.grid_shape[axis] + 1,
            )
            for axis in range(self.d)
        ]
        self.grid_coords = [0.5 * (e[:-1] + e[1:]) for e in self.grid_edges]
        self.grid_points = None
        self.store = store

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
        grid_points = self.get_grid_points()
        log_prob = log_prob_func(grid_points)
        prob = torch.exp(log_prob)
        prob = torch.reshape(prob, self.grid_shape)
        x = sample_hist(prob, bin_edges=self.grid_edges, n=n, noise=self.noise)
        x = self.send(x)
        return x

    def to(self, device):
        self.device = device
        self.grid_edges = [self.send(e) for e in self.grid_edges]
        self.grid_coords = [self.send(c) for c in self.grid_coords]
        return self


class SliceGridSampler:
    def __init__(
        self,
        grid_limits: List[Tuple[float]],
        grid_shape: Tuple[int],
        proj_dim: int = 2,
        int_size: int = 10000,
        int_method: str = "grid",
        int_batches: int = 1,
        noise: float = 0.0,
        verbose: bool = False,
        device=None,
    ):
        self.device = device
        self.grid_shape = grid_shape
        self.grid_limits = grid_limits
        self.dim = len(grid_limits)
        self.proj_dim = proj_dim
        self.samp_dim = self.dim - self.proj_dim
        self.noise = noise
        self.verbose = verbose

        self.grid_edges = [
            torch.linspace(
                self.grid_limits[axis][0],
                self.grid_limits[axis][1],
                self.grid_shape[axis] + 1,
            )
            for axis in range(self.dim)
        ]
        self.grid_coords = [0.5 * (e[:-1] + e[1:]) for e in self.grid_edges]

        # Projection grid.
        self.proj_axis = tuple(range(self.proj_dim))
        self.proj_grid_shape = self.grid_shape[: self.proj_dim]
        self.proj_grid_edges = self.grid_edges[: self.proj_dim]
        self.proj_grid_coords = self.grid_coords[: self.proj_dim]
        self.proj_grid_points = get_grid_points(*self.proj_grid_coords)

        # Sampling grid.
        self.samp_axis = tuple(range(self.proj_dim, self.dim))
        self.samp_grid_shape = self.grid_shape[self.proj_dim :]
        self.samp_grid_edges = self.grid_edges[self.proj_dim :]
        self.samp_grid_coords = self.grid_coords[self.proj_dim :]
        self.samp_grid_points = get_grid_points(*self.samp_grid_coords)

        # Integration limits (integration axis = sampling axis).
        self.int_size = int_size
        self.int_method = int_method
        self.int_batches = int_batches
        self.int_limits = self.grid_limits[self.proj_dim :]
        self.int_axis = self.samp_axis
        self.int_dim = self.samp_dim

        # We will evaluate the function on the sampling grid. The first `proj_dim`
        # dimensions are the projected coordinates.
        self.eval_points = torch.zeros((math.prod(self.samp_grid_shape), self.dim))
        self.eval_points[:, self.samp_axis] = get_grid_points(*self.samp_grid_coords)

    def to(self, device):
        self.device = device
        self.grid_edges = [self.send(e) for e in self.grid_edges]
        self.grid_coords = [self.send(c) for c in self.grid_coords]
        return self

    def send(self, x):
        return x.type(torch.float32).to(self.device)

    def project(self, func: Callable) -> torch.Tensor:
        """Project function onto onto lower dimensional plane."""

        x = torch.zeros((self.int_size, self.dim))

        def set_int_points(x):
            if self.int_method == "grid":
                int_res = self.int_size ** (1.0 / self.int_dim)
                int_res = math.ceil(int_res)
                int_res = int(int_res)
                int_coords = [
                    torch.linspace(xmin, xmax, int_res)
                    for (xmin, xmax) in self.int_limits
                ]
                int_points = get_grid_points(*int_coords)
                x[:, self.int_axis] = int_points[: x.shape[0], :]
            elif self.int_method == "uniform":
                for axis, (xmin, xmax) in zip(self.int_axis, self.int_limits):
                    x[:, axis] = random_uniform(xmin, xmax, self.int_size)
            elif self.int_method == "gaussian":
                scale = [xmax for (xmin, xmax) in self.int_limits]
                for axis, (xmin, xmax) in zip(self.int_axis, self.int_limits):
                    x[:, axis] = 0.75 * (xmax - xmin) * torch.randn(self.int_size)
            else:
                raise ValueError("Invalid int_method")
            return x

        rho = torch.zeros(self.proj_grid_points.shape[0])
        for _ in range(self.int_batches):
            x = set_int_points(x)
            for i in tqdm_wrapper(range(rho.shape[0]), self.verbose):
                x[:, self.proj_axis] = self.proj_grid_points[i, :]
                rho[i] += torch.sum(func(x))
        rho = torch.reshape(rho, self.proj_grid_shape)
        return rho

    def __call__(self, log_prob_func: Callable, n: int) -> torch.Tensor:
        # Compute projection.
        if self.verbose:
            print("Projecting")

        proj = self.project(lambda x: torch.exp(log_prob_func(x)))
        proj = proj / torch.sum(proj)

        # Resample projection. This determines the number of particles to place in each cell.
        n_samples_per_cell = resample_hist(proj, n)

        # Sample within each slice of the projection axis.
        if self.verbose:
            print("Sampling")

        x = []
        for indices in tqdm_wrapper(np.ndindex(*proj.shape), self.verbose):
            size = int(n_samples_per_cell[indices])
            if size == 0:
                continue

            # Sample the projection coordinates from a uniform distribution over
            # the current cell in the projected space.
            y = torch.zeros((size, self.dim))
            for axis, index in enumerate(indices):
                y[:, axis] = random_uniform(
                    self.proj_grid_edges[axis][index],
                    self.proj_grid_edges[axis][index + 1],
                    size=size,
                )
                if self.noise:
                    delta = (
                        self.proj_grid_edges[axis][index + 1]
                        - self.proj_grid_edges[axis][index]
                    )
                    y[:, axis] += self.noise * delta * (torch.rand(size) - 0.5)

            # Set the evaluation points on the projection axis.
            for axis, index in enumerate(indices):
                self.eval_points[:, axis] = self.proj_grid_coords[axis][index]

            # Grid sample.
            prob = torch.exp(log_prob_func(self.eval_points))
            prob = torch.reshape(prob, self.samp_grid_shape)
            y[:, self.samp_axis] = sample_hist(
                prob, self.samp_grid_edges, n=size, noise=self.noise
            )

            x.append(y)

        x = torch.vstack(x)
        x = x[torch.randperm(x.shape[0])]
        return x
