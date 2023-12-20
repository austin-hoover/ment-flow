"""Iterative maximum-entropy tomography (MENT) solver.

References
----------
[1] https://doi.org/10.1109/78.80970
[2] https://doi.org/10.1103/PhysRevAccelBeams.25.042801
"""
import abc
import typing
from typing import Any
from typing import Callable
from typing import Iterable
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import scipy.interpolate
import torch

from mentflow.sample import GridSampler
from mentflow.utils import get_grid_points_torch
from mentflow.utils import grab


class GaussianPrior:
    """Gaussian prior distribution."""
    def __init__(self, d=2, scale=1.0, device=None):
        mean = torch.zeros(d)
        mean = mean.type(torch.float32)
        if device is not None:
            mean = mean.to(device)

        cov = torch.eye(d) * (scale ** 2)
        cov = cov.type(torch.float32)
        if device is not None:
            cov = cov.to(device)
            
        self._dist = torch.distributions.MultivariateNormal(mean, cov)

    def log_prob(self, x):
        return self._dist.log_prob(x)


class UniformPrior:
    """Uniform prior distribution."""
    def __init__(self, d=2, scale=10.0, device=None):
        self.scale = scale
        self.d = d
        self.volume = (100.0) ** self.d
        self.device = device

    def log_prob(self, x):
        _log_prob = np.log(1.0 / self.volume)
        _log_prob = torch.ones(x.shape[0]) * _log_prob
        _log_prob = _log_prob.type(torch.float32)
        if self.device is not None:
            _log_prob = _log_prob.to(self.device)
        return _log_prob


class MENT_2D1D:
    """2D MENT reconstruction from 1D projections."""
    def __init__(
        self,
        transforms: List[Callable],
        diagnostic: Callable,
        measurements: List[torch.Tensor],
        prior: Any = None,
        interpolate: Union[bool, str] = "pchip",
        sampler=None,
        device=None,
    ) -> None:
        """Constructor.

        Parameters
        ----------
        transforms : list[callable]
            `transforms[i](x)` maps the input coordinates x to the ith measurement.
            Nonlinear maps are possible but we must track the Jacobian determinant;
            see comment in code below.
        diagnostic : callable
            `diagnostic(x)` generates the measurement data.
        measurements : list[ndarray]
            A list of measured one-dimensional projections. They should be normalized.
        prior : object
            The prior distribution. Must implement `prob(x)`. Defaults to uniform
            distribution with wide support [-100, 100].
        interpolate : bool, str
            Interpolation method when evaluating the component functions ("h" functions).
            If False, or None, use the discretized version of the functions. This necessarily
            results in uniform-density polygons in the reconstructed distribution.
            Options:
                - "linear": linear interpolation (np.interp)
                - "pchip": monotonic cubic splines (scipy.interpolate.PchipInterpolator)
        sampler : Sampler
            Implements `sample(f, n)` method to sample points from distribution function f.
        """
        self.d = 2
        self.iteration = 0
        self.interpolate = interpolate
        self.transforms = transforms
        self.diagnostic = self.set_diagnostic(diagnostic)
        self.measurements = self.set_measurements(measurements)

        self.prior = prior
        if self.prior is None:
            self.prior = UniformPrior(scale=100.0)

        self.lagrange_multipliers = self.initialize_lagrange_multipliers()

        self.sampler = sampler
        if self.sampler is None:
            xmax = 1.5 * max(self.bin_edges)
            limits = 2 * [(-xmax, +xmax)]
            self.sampler = GridSampler(limits=limits, res=200)

        self.device = device
        if self.device is None:
            self.device = torch.device("cpu")

    def _send(self, x):
        return x.type(torch.float32).to(self.device)

    def set_diagnostic(self, diagnostic: Callable):
        """Set the diagnostic (histogrammer)."""
        self.diagnostic = diagnostic
        if self.diagnostic is not None:
            self.n_bins = len(self.diagnostic.bin_edges)
            self.bin_edges = self.diagnostic.bin_edges
            self.bin_coords = 0.5 * (self.bin_edges[:-1] + self.bin_edges[1:])
        return self.diagnostic

    def set_measurements(self, measurements: List[torch.Tensor]):
        """Set the measurement list."""
        self.measurements = measurements
        if self.measurements is None:
            self.measurements = []
        self.n_meas = self.n_constraints = len(self.measurements)
        return self.measurements

    def initialize_lagrange_multipliers(self):
        """Initialize the model to the prior distribution."""
        self.lagrange_multipliers = []
        for i, measurement in enumerate(self.measurements):
            self.lagrange_multipliers.append([])
            for j in range(len(measurement)):
                g_ij = float(measurement[j] > 0)
                self.lagrange_multipliers[-1].append(g_ij)
        return self.lagrange_multipliers

    def _chi(self, x: torch.Tensor, i: int = 0, j: int = 0) -> torch.Tensor:
        """Return 1 if x is inside the jth bin of the ith projection; 0 otherwise."""
        idx = torch.logical_and(
            x[:, 0] >= self.bin_edges[j],
            x[:, 0] < self.bin_edges[j + 1],
        )
        return idx.float()

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """Return the logarithm of the probability density at x.

        See Dusaussoy Eq. (15).
        """
        log_prob = torch.ones(len(x))
        log_prob = self._send(log_prob)
        for i, transform in enumerate(self.transforms):
            u = transform(x)
            h_func = np.ones(len(u))
            if self.interpolate:
                points = grab(self.bin_coords)
                values = self.lagrange_multipliers[i]
                int_points = grab(u[:, 0])
                if self.interpolate == "linear":
                    h_func = np.interp(int_points, points, values)
                elif self.interpolate == "pchip":
                    fint = scipy.interpolate.PchipInterpolator(points, values, axis=0, extrapolate=None)
                    h_func = fint(int_points)
                else:
                    raise ValueError("Invalid `interpolate` method.")
            else:
                for j in range(len(self.measurements[i])):
                    h_func += grab(self._chi(u, i, j)) * self.lagrange_multipliers[i][j]
            h_func = torch.tensor(h_func)
            h_func = self._send(h_func)
            log_prob += torch.log(h_func)
        log_prob += self.prior.log_prob(x)
        return log_prob

    def prob(self, x: torch.Tensor) -> torch.Tensor:
        """Return the probability density at x."""
        return torch.exp(self.log_prob(x))

    def sample(self, n: int) -> torch.Tensor:
        """Sample particles from the distribution."""
        x = self.sampler(log_prob_func=self.log_prob, n=n)
        x = self._send(x)
        return x

    def sample_and_log_prob(self, n: int) -> Tuple[np.ndarray, np.ndarray]:
        """Sample particles x and compute log(prob(x))."""
        x = self.sample(n)
        log_prob = self.log_prob(x)
        return (x, log_prob)

    def _normalize_projection(self, projection: np.ndarray) -> np.ndarray:
        """Normalize one-dimensional projection."""
        bin_volume = self.bin_edges[1] - self.bin_edges[0]
        normalization = projection.sum() * bin_volume
        return projection / normalization

    def _simulate_integrate(self, index: int = 0, xmax: float = None, res: int = 150) -> torch.Tensor:
        """Compute the ith projection using numerical integration.

        This function evaluates the density on a grid in the transformed space using
        the Jacobian determinant of the transfer map, then integrates the density to
        obtain the projection onto the measurement axis.

        Parameters
        ----------
        index : int
            The measurement index.
        xmax : float
            Defines the integration range in the transformed momentum space (u'). Defaults to
            1.5 times the maximum measurement bin coordinate.
        res : int
            Number of points in the integration grid.

        Returns
        -------
        tensor, shape (len(self.bin_coords),)
            The projection onto the ith measurement axis.
        """
        # Compute the density on a regular grid in the transformed space.
        # We do this by flowing backward and tracking the Jacobian determinant
        # (determinant = 1 for linear transformations).
        if xmax is None:
            xmax = 1.5 * max(self.bin_coords)
        int_grid_xmax = xmax  # integration limits
        int_res = res  # integration grid resolution
        grid_shape = (len(self.bin_coords), int_res)
        grid_coords = [
            self.bin_coords,  # measurement axis
            torch.linspace(-int_grid_xmax, +int_grid_xmax, int_res),  # integration axis
        ]
        grid_coords = [self._send(c) for c in grid_coords]
        grid_points = get_grid_points_torch(grid_coords)
        grid_points = self._send(grid_points)
        transform = self.transforms[index]
        grid_points_in = transform.inverse(grid_points)
        prob_in = self.prob(grid_points_in)

        # Assume det(J) = 1 for now; in the future, we should add a `forward_and_log_det`
        # method to `transform` object to track the Jacobian determinant.
        prob_out = prob_in

        # Compute the projection in the transformed space.
        prob_out = prob_out.reshape(grid_shape)
        prediction = torch.sum(prob_out, dim=1)
        prediction = self._normalize_projection(prediction)
        return prediction

    def _simulate_sample(self, index: int = 0, n: int = 100000) -> torch.Tensor:
        """Compute the ith projection using particle tracking + density estimation.

        This function samples particles from the model distribution, tracks the
        particles using the ith transform function, and estimates the projected
        density using a histogram.

        Parameters
        ----------
        index : int
            The measurement index.
        n : int
            The number of particles to sample.

        Returns
        -------
        tensor, shape (len(self.bin_coords),)
            The projection onto the ith measurement axis.
        """
        transform = self.transforms[index]
        x = self.sample(n)
        x = self._send(x)
        u = transform(x)
        prediction = self.diagnostic(u)
        prediction = self._normalize_projection(prediction)
        return prediction

    def _simulate(self, index: int = 0, method: str = "integrate", **kws) -> torch.Tensor:
        """Compute the ith projection.

        Parameters
        ----------
        index : int
            The measurement index.
        method : {"integrate", "sample"}
            Method to compute the projections of the distribution. See `simulate` method.
            - "integrate":
                Compute the density on a regular grid in the transformed space; evaluate
                the density on the grid using Jacobian determinant of transfer map.
            - "particles":
                Sample particles from the distribution, track the particles using the
                specified transform function, and estimate the projected density using a
                histogram.
        **kws
            Key word arguments passed to `self._simulate_integrate` (if method="integrate")
            or `self._simulate_sample` (if method="sample").

        Returns
        -------
        tensor, shape (len(self.bin_coords),)
            The projection onto the ith measurement axis.
        """
        _functions = {
            "integrate": self._simulate_integrate,
            "sample": self._simulate_sample,
        }
        _function = _functions[method]
        return _function(index, **kws)

    def simulate(self, **kws) -> torch.Tensor:
        """Compute all projections. Same arguments as `self._simulate`."""
        return [self._simulate(index, **kws) for index in range(self.n_meas)]

    def gauss_seidel_iterate(self, *args, **kws) -> None:
        """Execute Gauss-Seidel iteration to update lagrange multipliers.

        Parameters
        ----------
        *args
            Arguments passed to `self._simulate`.
        **kws
            Key word arguments passed to `self._simulate`.
        """
        for i, (transform, measurement) in enumerate(zip(self.transforms, self.measurements)):
            prediction = self._simulate(index=i, **kws)
            for j in range(len(self.lagrange_multipliers[i])):
                g_meas = float(measurement[j])
                g_pred = float(prediction[j])
                if (g_meas != 0.0) and (g_pred != 0.0):
                    self.lagrange_multipliers[i][j] *= (g_meas / g_pred)
        self.iteration += 1

    def parameters(self):
        return

    def save(path):
        return

    def load(path):
        return


    
