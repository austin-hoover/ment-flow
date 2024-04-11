"""Differentiable diagnostics."""
from typing import Iterable
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import torch

from mentflow.diag.hist import kde_histogram_1d
from mentflow.diag.hist import kde_histogram_2d
from mentflow.utils import coords_from_edges
from mentflow.utils import grab
from mentflow.utils import unravel


class Diagnostic(torch.nn.Module):
    def __init__(self, device: torch.device = None, seed: int = None, ndim: int = None) -> None:
        super().__init__()
        self.device = device
        self.seed = seed
        self.ndim = ndim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class Histogram(Diagnostic):
    def __init__(
        self, 
        noise: bool = False,
        noise_scale: float = 0.0, 
        noise_type: str = "gaussian", 
        **kws
    ) -> None:
        super().__init__(**kws)
        self.noise = noise
        self.noise_scale = noise_scale
        self.noise_type = noise_type

    def set_noise(self, setting: bool) -> None:
        self.noise = setting

    def project(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
        
    def bin(self, x_proj: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError   

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hist = self.bin(self.project(x))

        if self.noise and self.noise_scale > 0.0:
            rng = torch.Generator(device=self.device)
            if self.seed is not None:
                rng.manual_seed(self.seed)
            if self.noise_type == "uniform":
                frac_noise = torch.rand(hist.shape[0], generator=rng, device=self.device) * 2.0
                frac_noise = frac_noise * self.noise_scale
            elif self.noise_type == "gaussian":
                frac_noise = torch.randn(hist.shape[0], generator=rng, device=self.device)
                frac_noise = frac_noise * self.noise_scale
            else:
                frac_noise = torch.zeros(hist.shape)
                frac_noise = frac_noise.type(torch.float32).to(self.device)
            hist = hist * (1.0 + frac_noise)
            hist = torch.clamp(hist, 0.0, None)
        return hist
            

class Histogram1D(Histogram):
    def __init__(
        self,
        edges: torch.Tensor,
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
        edges : tensor
            Histogram bin edges.
        bandwidth : float
            Gaussian kernel width relative to bin width (default: 1.0)
        direction : tensor
            If provided, the projection is computed along this vector. Otherwise
            use `axis`.
        kde : bool
            Turns on/off KDE.
        **kws
            Key word arguments passed to `Histogram`.
        """
        super().__init__(**kws)
        self.axis = axis
        self.kde = kde
        self.ndim = 1

        self.direction = direction
        if self.direction is not None:
            self.direction = self.direction / torch.norm(self.direction)

        if bandwidth is None:
            bandwidth = 0.5
        
        self.register_buffer("edges", edges)
        self.register_buffer("coords", coords_from_edges(self.edges))
        self.register_buffer("resolution", edges[1] - edges[0])           
        self.register_buffer("bandwidth", bandwidth * self.resolution)

    def project(self, x: torch.Tensor) -> torch.Tensor:
        x_proj = None
        if self.direction is None:
            x_proj = x[:, self.axis]
        else:
            x_proj = torch.sum(x * self.direction, dim=1)
        return x_proj

    def bin(self, x_proj: torch.Tensor) -> torch.Tensor:
        if self.kde:
            hist = kde_histogram_1d(x_proj, self.edges, bandwidth=self.bandwidth)
            return hist
        else:
            hist = torch.histogram(x_proj, self.edges, density=True)
            hist = hist.hist
            return hist


class Histogram2D(Histogram):
    def __init__(
        self,
        axis: Iterable[int],
        edges: Iterable[torch.Tensor],
        bandwidth: Iterable[torch.Tensor],
        kde: bool = True,
        **kws
    ) -> None:
        """Constructor.

        Parameters
        ----------
        axis : tuple(int, int)
            Histogram is computed along this axis.
        edges: list[tensor]
            Histogram bin edges along each axis.
        bandwidth : iterable[float]
            Gaussian kernel widths relative to bin widths (default = 0.5).
        kde : bool
            Turns on/off KDE.
        **kws
            Key word arguments passed to `Histogram`.
        """
        super().__init__(**kws)
        self.axis = axis
        self.kde = kde
        self.ndim = 2

        bandwidth_x, bandwidth_y = bandwidth
        if bandwidth_x is None:
            bandwidth_x = 0.5
        if bandwidth_y is None:
            bandwidth_y = 0.5
        
        self.register_buffer("edges_x", edges[0])
        self.register_buffer("edges_y", edges[1])
        self.register_buffer("coords_x", coords_from_edges(self.edges_x))
        self.register_buffer("coords_y", coords_from_edges(self.edges_y))
        self.register_buffer("resolution_x", self.edges_x[1] - self.edges_x[0])           
        self.register_buffer("resolution_y", self.edges_y[1] - self.edges_y[0])           
        self.register_buffer("bandwidth_x", bandwidth_x * self.resolution_x)  
        self.register_buffer("bandwidth_y", bandwidth_y * self.resolution_y)  
        
    def project(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, self.axis]
        
    def bin(self, x_proj: torch.Tensor) -> torch.Tensor:
        if self.kde:
            hist = kde_histogram_2d(
                x_proj[:, 0],
                x_proj[:, 1],
                bins=(self.edges_x, self.edges_y),
                bandwidth=(self.bandwidth_x, self.bandwidth_y),
            )
            return hist
        else:
            # MPS throws error for torch.histogramdd. Convert to numpy for now.
            # We only call this function during evaluation anyway.
            hist, _ = np.histogramdd(
                grab(x_proj),
                bins=[grab(self.edges_x), grab(self.edges_y)],
                density=True,
            )
            hist = torch.from_numpy(hist)
            hist = hist.type(torch.float32).to(self.device)
            return hist


class Projection(Diagnostic):
    """Projects points onto axis (no density estimation)."""
    def __init__(self, axis: Union[int, Tuple[int]], **kws) -> None:
        super().__init__(**kws)
        self.axis = axis

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, self.axis]


