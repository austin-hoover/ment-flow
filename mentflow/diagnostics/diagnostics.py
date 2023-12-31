"""Differentiable diagnostics."""
from typing import Iterable
from typing import Optional

import numpy as np
import torch

from mentflow.diagnostics.histogram import kde_histogram_1d
from mentflow.diagnostics.histogram import kde_histogram_2d
from mentflow.utils import centers_from_edges


class Diagnostic(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class Histogram1D(Diagnostic):
    """One-dimensional histogram."""
    def __init__(
        self,
        axis: int,
        bin_edges: torch.Tensor,
        bandwidth: Optional[float] = None,
        kde=True,
        **kws
    ) -> None:
        """Constructor.

        Parameters
        ----------
        axis : int
            Histogram is computed along this axis.
        bin_edges : tensor
            Histogram bin edges.
        bandwidth : float
            Gaussian kernel width relative to bin width (default: 1.0)
        **kws
            Key word arguments passed to `Diagnostic`.
        """
        super().__init__(**kws)
        self.axis = axis
        self.shape = len(bin_edges) - 1
        self.register_buffer("bin_edges", bin_edges)
        self.register_buffer("bin_coords", centers_from_edges(self.bin_edges))
        self.register_buffer("resolution", bin_edges[1] - bin_edges[0])
        if bandwidth is None:
            bandwidth = 1.0
        self.register_buffer("bandwidth", bandwidth * self.resolution)
        self.kde = kde

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Estimate probability density. 

        Parameters
        ----------
        x : tensor, shape (n, d)
            Phase space coordinate array.

        Returns
        -------
        hist : tensor
            The estimated probability density.
        """
        hist = None
        if self.kde:
            hist = kde_histogram_1d(x[:, self.axis], bin_edges=self.bin_edges, bandwidth=self.bandwidth)
            return hist
        else:
            hist = torch.histogram(x[:, self.axis], bins=self.bin_edges, density=True)
            hist = hist.hist
            return hist
            

class Histogram2D(Diagnostic):
    """Two-dimensional histogram."""
    def __init__(
        self,
        axis: Iterable[int],
        bin_edges: Iterable[torch.Tensor],
        bandwidth: Optional[Iterable[float]] = None,
        kde=True,
        **kws
    ) -> None:
        """Constructor.

        Parameters
        ----------
        axis : tuple(int, int)
            Histogram is computed along this axis.
        bin_edges : list[tensor, tensor]
            Histogram bin edges.
        bandwidth : list[float]
            Gaussian kernel width relative to bin widths (default: 1.0).
        **kws
            Key word arguments passed to `Diagnostic`.
        """
        super().__init__(**kws)
        self.axis = axis
        self.shape = tuple([(len(e) - 1) for e in bin_edges])
        self.register_buffer("bin_edges_x", bin_edges[0])
        self.register_buffer("bin_edges_y", bin_edges[1])
        self.register_buffer("bin_coords_x", centers_from_edges(bin_edges[0]))
        self.register_buffer("bin_coords_y", centers_from_edges(bin_edges[1]))
        self.register_buffer("resolution_x", self.bin_edges_x[1] - self.bin_edges_x[0])
        self.register_buffer("resolution_y", self.bin_edges_y[1] - self.bin_edges_y[0])

        d = len(axis)
        if bandwidth is None:
            bandwidth = d * [1.0]
        if type(bandwidth) in [float, int]:
            bandwidth = d * [bandwidth]
        self.register_buffer("bandwidth_x", bandwidth[0] * self.resolution_x)
        self.register_buffer("bandwidth_y", bandwidth[1] * self.resolution_y)
        self.kde = kde

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Estimate probability density. 

        Parameters
        ----------
        b : tensor, shape (n, d)
            Phase space coordinate array.

        Returns
        -------
        hist : tensor
            The estimated probability density.
        """
        if self.kde:
            hist = kde_histogram_2d(
                x[:, self.axis[0]],
                x[:, self.axis[1]],
                bin_edges=[self.bin_edges_x, self.bin_edges_y],
                bandwidth=[self.bandwidth_x, self.bandwidth_y],
            )
            return hist
        else:
            hist = torch.histogramdd(
                x[:, self.axis],
                bins=[self.bin_edges_x, self.bin_edges_y],
                density=True,
            )
            hist = hist.hist
            return hist


class Projection(Diagnostic):
    """Projects points onto axis (no density estimation)."""
    def __init__(self, axis=0, **kws) -> None:
        super().__init__(**kws)
        self.axis = axis

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, self.axis]


