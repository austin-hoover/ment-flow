"""Differentiable diagnostics."""
from typing import Iterable
from typing import Optional

import numpy as np
import torch

from mentflow.diag.hist import kde_histogram_1d
from mentflow.diag.hist import kde_histogram_2d
from mentflow.utils import coords_from_edges
from mentflow.utils import unravel


class Diagnostic(torch.nn.Module):
    def __init__(self, device=None, seed=None) -> None:
        super().__init__()
        self.device = device
        self.seed = seed

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class Histogram(Diagnostic):
    def __init__(self, noise_scale=0.0, noise_type="gaussian", **kws):
        super().__init__(**kws)
        self.noise = True
        self.noise_scale = noise_scale
        self.noise_type = noise_type

    def set_noise(self, setting) -> None:
        self.noise = setting
        
    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hist = self._forward(x)

        if self.noise and self.noise_scale:
            generator = torch.Generator(device=self.device)
            if self.seed is not None:
                generator.manual_seed(self.seed)
            if self.noise_type == "uniform":
                frac_noise = torch.rand(hist.shape[0], generator=generator, device=self.device) * 2.0
                frac_noise = frac_noise * self.noise_scale
            elif self.noise_type == "gaussian":
                frac_noise = torch.randn(hist.shape[0], generator=generator, device=self.device)
                frac_noise = frac_noise * self.noise_scale
            else:
                frac_noise = torch.zeros(hist.shape)
                frac_noise = frac_noise.type(torch.float32).to(self.device)
            hist = hist * (1.0 + frac_noise)
            hist = torch.clamp(hist, 0.0, None)
        return hist
            

class Histogram1D(Histogram):
    """One-dimensional histogram."""
    def __init__(
        self,
        bin_edges: torch.Tensor,
        bandwidth: Optional[float] = None,
        axis: int = 0,
        direction: torch.Tensor = None,
        kde: bool = True,
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
        direction : tensor
            If provided, the projection is computed along this vector. Otherwise
            use `axis`.
        **kws
            Key word arguments passed to `Histogram`.
        """
        super().__init__(**kws)
        self.axis = axis
        self.shape = len(bin_edges) - 1
        self.register_buffer("bin_edges", bin_edges)
        self.register_buffer("bin_coords", coords_from_edges(self.bin_edges))
        self.register_buffer("resolution", bin_edges[1] - bin_edges[0])        
        if bandwidth is None:
            bandwidth = 0.5
        self.register_buffer("bandwidth", bandwidth * self.resolution)
        self.kde = kde
        self.direction = direction
        if self.direction is not None:
            self.direction = self.direction / torch.norm(self.direction)

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
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
        data = None
        if self.direction is None:
            data = x[:, self.axis]
        else:
            data = torch.sum(x * self.direction, dim=1)
        if self.kde:
            hist = kde_histogram_1d(data, bin_edges=self.bin_edges, bandwidth=self.bandwidth)
            return hist
        else:
            hist = torch.histogram(data, bins=self.bin_edges, density=True)
            hist = hist.hist
            return hist


class Histogram2D(Histogram):
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
            Key word arguments passed to `Histogram`.
        """
        super().__init__(**kws)
        self.axis = axis
        self.shape = tuple([(len(e) - 1) for e in bin_edges])
        self.bin_edges = bin_edges
        self.bin_coords = [coords_from_edges(e) for e in bin_edges]
        self.resolution = [e[1] - e[0] for e in bin_edges]

        self.d = d = len(axis)
        self.bandwidth = bandwidth
        if self.bandwidth  is None:
            self.bandwidth  = d * [0.5]
        if type(self.bandwidth ) in [float, int]:
            self.bandwidth  = d * [self.bandwidth]
        self.bandwidth = [self.bandwidth [i] * self.resolution[i] for i in range(d)]
        self.kde = kde

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
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
                bin_edges=self.bin_edges,
                bandwidth=self.bandwidth,
            )
            return hist
        else:
            hist = torch.histogramdd(
                x[:, self.axis],
                bins=self.bin_edges,
                density=True,
            )
            hist = hist.hist
            return hist

    def to(self, device):
        for i in range(self.d):
            self.bin_edges[i] = self.bin_edges[i].to(device)
            self.bin_coords[i] = self.bin_coords[i].to(device)
        self.device = device
        return self


class Projection(Diagnostic):
    """Projects points onto axis (no density estimation)."""
    def __init__(self, axis=0, **kws) -> None:
        super().__init__(**kws)
        self.axis = axis

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, self.axis]


