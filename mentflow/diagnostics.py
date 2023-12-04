"""Differentiable diagnostics.

KDE code provided by Ryan Roussel (https://link.aps.org/doi/10.1103/PhysRevLett.130.145001).
"""
from typing import Iterable
from typing import List
from typing import Optional
from typing import Tuple
import numpy as np
import torch
import torch.nn as nn

import mentflow.utils as utils


def marginal_pdf(
    values: torch.Tensor, 
    coords: torch.Tensor, 
    sigma: float = 1.0, 
    epsilon: float = 1.00e-10,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Estimate one-dimensional probability density function from samples.

    Parameters
    ----------
    values : tensor, shape (n,)
        One-dimensional data points.
    coords : tensor, shape (k,)
        Grid coordinates.
    sigma : float
        Gaussian kernel width.
    epsilon : float
        Padding for normalization.

    Returns
    -------
    prob : tensor, shape (k,)
        Estimated probability at each point in `coords`.
    kernel_values : tensor, shape (n, k)
        Kernel value at each point.
    """
    residuals = values - coords.repeat(*values.shape)
    kernel_values = torch.exp(-0.5 * (residuals / sigma).pow(2))
    prob = torch.mean(kernel_values, dim=-2)
    delta = coords[1] - coords[0]
    normalization = torch.sum(prob * delta)
    normalization = normalization + epsilon
    prob = prob / normalization
    return (prob, kernel_values)


def joint_pdf(
    kernel_values_x: torch.Tensor, 
    kernel_values_y: torch.Tensor, 
    coords: Iterable[torch.Tensor],
    epsilon: float = 1.00e-10,
) -> torch.Tensor:
    """Estimate two-dimensional probability density function from samples.

    Parameters
    ----------
    kernel_values_x, kernel_values_y : 
        Kernel values for x and y.
    coords : list[torch.Tensor, shape (k,)], shape (2,)
        Regular grid coordinates.
    epsilon : float
        Padding for normalization.

    Returns
    -------
    prob : tensor, shape (len(coords[0]), len(coords[1]))
        Estimated probability on the grid.
    """
    prob = torch.matmul(kernel_values_x.transpose(-2, -1), kernel_values_y)
    delta = [c[1] - c[0] for c in coords]
    normalization = torch.sum(prob * delta[0] * delta[1])
    normalization = normalization + epsilon
    prob = prob / normalization
    return prob


def histogram(
    x: torch.Tensor, 
    bin_edges: torch.Tensor, 
    bandwidth: float = 1.0, 
    epsilon: float = 1.00e-10
) -> torch.Tensor:
    """Estimate one-dimensional histogram using kernel density estimation."""
    coords = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    prob, _ = marginal_pdf(x.unsqueeze(-1), coords, bandwidth, epsilon)
    return prob


def histogram2d(
    x: torch.Tensor,
    y: torch.Tensor,
    bin_edges: Iterable[torch.Tensor],
    bandwidth: Iterable[float] = (1.0, 1.0),
    epsilon: float = 1.00e-10,
) -> torch.Tensor:
    """Estimate two-dimensional histogram using kernel density estimation."""
    coords = [0.5 * (e[:-1] + e[1:]) for e in bin_edges]
    _, kernel_values_x = marginal_pdf(x.unsqueeze(-1), coords[0], bandwidth[0], epsilon)
    _, kernel_values_y = marginal_pdf(y.unsqueeze(-1), coords[1], bandwidth[1], epsilon)
    prob = joint_pdf(kernel_values_x, kernel_values_y, coords, epsilon=epsilon)
    return prob


class Diagnostic(torch.nn.Module):
    """Represents a measurement device."""
    def __init__(self, transform: Optional[torch.nn.Module] = None) -> None:
        """Constructor.

        Parameters
        ----------
        transform : torch.nn.Module
            Transformation applied to the particles before measurement. Defaults
            to the identity transformation.
        """
        super().__init__()
        self.transform = transform
        if self.transform is None:
            self.transform = torch.nn.Identity()

    def forward(self, x, **kws):
        """Transform, then measure the particle distribution."""
        return self._forward(self.transform(x), **kws)

    def _forward(self, x, **kws):
        """Measure the particle distribution."""
        raise NotImplementedError


class Histogram1D(Diagnostic):
    """One-dimensional histogram diagnostic."""
    def __init__(
        self,
        axis: int,
        bin_edges: torch.Tensor,
        bandwidth: Optional[float] = None,
    ) -> None:
        """Constructor.

        Parameters
        ----------
        axis : int
            Histogram is computed along this axis.
        bin_edges : tensor
            Histogram bin edges.
        bandwidth : float
            Gaussian kernel width (default is bin width).
        """
        super().__init__()
        self.axis = axis
        self.register_buffer("bin_edges", bin_edges)
        self.register_buffer("bin_centers", utils.centers_from_edges(self.bin_edges))
        self.register_buffer("resolution", bin_edges[1] - bin_edges[0])
        self.register_buffer("bandwidth", self.resolution if bandwidth is None else bandwidth)

    def _forward(self, x, kde=True):
        """Estimate probability density. 

        Parameters
        ----------
        x : tensor, shape (n, d)
            Phase space coordinate array.
        kde : bool
            Whether to use kernel density estimation or regular binning. (KDE is differentiable.)

        Returns
        -------
        hist : tensor
            The estimated probability density.
        """
        hist = None
        if kde:
            hist = histogram(x[:, self.axis], bin_edges=self.bin_edges, bandwidth=self.bandwidth)
            return hist
        else:
            hist = torch.histogram(x[:, self.axis], bins=self.bin_edges, density=True)
            hist = hist.hist
            return hist
            

class Histogram2D(Diagnostic):
    """Two-dimensional histogram diagnostic."""
    def __init__(
        self,
        axis: Iterable[int],
        bin_edges: Iterable[torch.Tensor],
        bandwidth: Optional[Iterable[float]] = None,
    ) -> None:
        """Constructor.

        Parameters
        ----------
        axis : tuple(int, int)
            Histogram is computed along this axis.
        bin_edges : list[tensor, tensor]
            Histogram bin edges.
        bandwidth : float
            Gaussian kernel width (default is bin width).
        """
        super().__init__()
        self.axis = axis
        self.register_buffer("bin_edges", torch.nested.nested_tensor(bin_edges))
        self.register_buffer(
            "bin_centers",
            torch.nested.nested_tensor(
                [utils.centers_from_edges(self.bin_edges[i]) for i in range(len(self.axis))]
            ),
        )

        d = len(axis)
        resolution = torch.zeros(d)
        for i in range(d):
            resolution[i] = self.bin_edges[i][1] - self.bin_edges[i][0]
        self.register_buffer("resolution", resolution)

        if bandwidth is None:
            bandwidth = d * [None]
        for i in range(d):
            bandwidth[i] = bandwidth[i] if bandwidth[i] else self.resolution[i]
        self.register_buffer("bandwidth", torch.tensor(bandwidth))

    def _forward(self, x, kde=True):
        """Estimate probability density. 

        Parameters
        ----------
        x : tensor, shape (n, d)
            Phase space coordinate array.
        kde : bool
            Whether to use kernel density estimation or regular binning. (KDE is differentiable.)

        Returns
        -------
        hist : tensor
            The estimated probability density.
        """
        if kde:
            hist = histogram2d(
                x[:, self.axis[0]],
                x[:, self.axis[1]],
                bin_edges=self.bin_edges,
                bandwidth=self.bandwidth,
            )
            return hist
        else:
            hist = torch.histogramdd(
                x[:, self.axis],
                bins=[self.bin_edges[i] for i in range(self.bin_edges.size(0))],
                density=True,
            )
            hist = hist.hist
            return hist
