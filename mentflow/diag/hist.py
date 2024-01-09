"""Differentiable kernel density estimation (KDE).

Code provided by R. Roussel (https://link.aps.org/doi/10.1103/PhysRevLett.130.145001).
"""
from typing import Iterable
from typing import Tuple

import torch


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


def kde_histogram_1d(
    x: torch.Tensor, 
    bin_edges: torch.Tensor, 
    bandwidth: float = 1.0, 
    epsilon: float = 1.00e-10
) -> torch.Tensor:
    """Estimate one-dimensional histogram using kernel density estimation."""
    coords = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    prob, _ = marginal_pdf(x.unsqueeze(-1), coords, bandwidth, epsilon)
    return prob
    

def kde_histogram_2d(
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

