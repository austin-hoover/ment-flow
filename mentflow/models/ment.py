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
from mentflow.utils import get_grid_points
from mentflow.utils import grab


class GaussianPrior:
    """Gaussian prior distribution."""
    def __init__(self, scale=1.0):
        self.scale = scale

    def prob(self, x):
        return np.exp(-(1.0 / (2.0 * self.scale**2)) * np.sum(np.square(x), axis=1)) / (
            self.scale * np.sqrt(2.0 * np.pi)
        )


class UniformPrior:
    """Uniform prior distribution."""
    def __init__(self, scale=100.0, d=2):
        self.scale = scale
        self.d = d
        self.volume = (100.0) ** self.d

    def prob(self, x):
        return np.full(len(x), 1.0 / self.volume)


class LinearTransform:
    """Linear transformation."""
    def __init__(self, matrix):
        self.matrix = matrix
        self.matrix_inv = np.linalg.inv(matrix)

    def forward(self, x):
        return np.matmul(x, self.matrix.T)

    def inverse(self, x):
        return np.matmul(x, self.matrix_inv.T)

    def __call__(self, x):
        return self.forward(x)


class HistogramDiagnostic:
    """One-dimensional histogram diagnostic."""
    def __init__(self, bin_edges, axis=0):
        self.bin_edges = bin_edges
        self.axis = axis

    def __call__(self, x):
        hist, _ = np.histogram(x[:, self.axis], self.bin_edges, density=True)
        return hist


class MENT_2D1D:
    """2D MENT reconstruction from 1D projections. [NumPy version]"""
    def __init__(
        self,
        transforms: List[Callable],
        diagnostic: Callable,
        measurements: List[np.ndarray],
        prior: Any = None,
        interpolate: Union[bool, str] = "pchip",
        sampler=None,
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
            xmax = 1.5 * np.max(self.bin_edges)
            limits = 2 * [(-xmax, +xmax)]
            self.sampler = GridSampler(limits=limits, res=200)

    def set_diagnostic(self, diagnostic: Callable):
        """Set the diagnostic (histogrammer)."""
        self.diagnostic = diagnostic
        if self.diagnostic is not None:
            self.n_bins = self.diagnostic.bin_edges
            self.bin_edges = self.diagnostic.bin_edges
            self.bin_coords = 0.5 * (self.bin_edges[:-1] + self.bin_edges[1:])
        return self.diagnostic

    def set_measurements(self, measurements: List[np.ndarray]):
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
            self.lagrange_multipliers[-1] = np.array(self.lagrange_multipliers[-1])
        return self.lagrange_multipliers

    def _chi(self, x: np.ndarray, i: int = 0, j: int = 0) -> np.ndarray:
        """Return 1 if x is inside the jth bin of the ith projection; 0 otherwise."""
        idx = np.logical_and(
            x[:, 0] >= self.bin_edges[j],
            x[:, 0] < self.bin_edges[j + 1],
        )
        return idx.astype(float)

    def prob(self, x: np.ndarray) -> np.ndarray:
        """Return the probability density at x.

        See Dusaussoy Eq. (15).
        """
        _prob = np.ones(len(x))
        for i, transform in enumerate(self.transforms):
            u = transform(x)
            h_func = np.zeros(len(u))
            if self.interpolate:
                points = self.bin_coords
                values = self.lagrange_multipliers[i][:]
                int_points = u[:, 0]
                if self.interpolate == "linear":
                    h_func = np.interp(int_points, points, values)
                elif self.interpolate == "pchip":
                    fint = scipy.interpolate.PchipInterpolator(
                        points, values, axis=0, extrapolate=None
                    )
                    h_func = fint(int_points)
                else:
                    raise ValueError("Invalid `interpolate` method.")
            else:
                for j in range(len(self.measurements[i])):
                    h_func += self._chi(u, i, j) * self.lagrange_multipliers[i][j]
            _prob *= h_func
        return _prob * self.prior.prob(x)

    def log_prob(self, x: np.ndarray) -> np.ndarray:
        """Return the logarithm of the probability density at x."""
        return np.log(self.prob(x))

    def sample(self, n: int) -> np.ndarray:
        """Sample particles from the distribution."""
        x = self.sampler(func=self.prob, n=n)
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

    def _simulate_integrate(self, index: int = 0, xmax: float = None, res: int = 150) -> np.ndarray:
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
        ndarray, shape (len(self.bin_coords),)
            The projection onto the ith measurement axis.
        """
        # Compute the density on a regular grid in the transformed space.
        # We do this by flowing backward and tracking the Jacobian determinant
        # (determinant = 1 for linear transformations).
        if xmax is None:
            xmax = 1.5 * np.max(self.bin_coords)
        int_grid_xmax = xmax  # integration limits
        int_res = res  # integration grid resolution
        grid_shape = (len(self.bin_coords), int_res)
        grid_coords = [
            self.bin_coords,  # measurement axis
            np.linspace(-int_grid_xmax, +int_grid_xmax, int_res),  # integration axis
        ]
        grid_points = get_grid_points(grid_coords)
        transform = self.transforms[index]
        grid_points_in = transform.inverse(grid_points)
        prob_in = self.prob(grid_points_in)

        # Assume det(J) = 1 for now; in the future, we should add a `forward_and_log_det`
        # method to `transform` object to track the Jacobian determinant.
        prob_out = prob_in

        # Compute the projection in the transformed space.
        prob_out = np.reshape(prob_out, grid_shape)
        prediction = np.sum(prob_out, axis=1)
        prediction = self._normalize_projection(prediction)
        return prediction

    def _simulate_sample(self, index: int = 0, n: int = 100000) -> np.ndarray:
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
        ndarray, shape (len(self.bin_coords),)
            The projection onto the ith measurement axis.
        """
        transform = self.transforms[index]
        x = self.sample(n)
        u = transform(x)
        prediction = self.diagnostic(u)
        prediction = self._normalize_projection(prediction)
        return prediction

    def _simulate(self, index: int = 0, method: str = "integrate", **kws) -> np.ndarray:
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
        ndarray, shape (len(self.bin_coords),)
            The projection onto the ith measurement axis.
        """
        _functions = {
            "integrate": self._simulate_integrate,
            "sample": self._simulate_sample,
        }
        _function = _functions[method]
        return _function(index, **kws)

    def simulate(self, **kws) -> np.ndarray:
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
                g_meas = measurement[j]
                g_pred = prediction[j]
                if (g_meas != 0.0) and (g_pred != 0.0):
                    factor = g_meas / g_pred
                    if factor > 10.0:
                        print("factor = {}".format(factor))
                    self.lagrange_multipliers[i][j] *= factor
        self.iteration += 1


class MENT_2D1D_Torch:
    """2D MENT reconstruction from 1D projections. [Torch version].

    Everything should be the same as `MENT_2D1D` with torch.Tensor instead 
    of np.ndarray
    """
    def __init__(self, **kws) -> None:
        prior = kws.get("prior", None)
        if prior is None:
            prior = torch.distributions.Uniform(
                -50.0 * torch.ones(2),
                +50.0 * torch.ones(2),
            )
        return super().__init__(prior=prior, **kws)

    def _chi(self, x: torch.Tensor, i: int = 0, j: int = 0) -> np.ndarray:
        idx = super()._chi(grab(x), i, j)
        return torch.from_numpy(idx)

    def prob(self, x: torch.Tensor) -> torch.Tensor:
        _prob = super().prob(grab(x))
        _prob = torch.from_numpy(_prob)
        return _prob
        
    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        return torch.log(self.prob(x))

    def sample(self, n: int) -> torch.Tensor:
        x = self.sampler(func=super().prob, n=n)
        x = torch.from_numpy(x)
        return x

    def _simulate_integrate(self, index: int = 0, xmax: float = None, res: int = 150) -> np.ndarray:
        if xmax is None:
            xmax = 1.5 * max(self.bin_coords)
        int_grid_xmax = xmax
        int_res = res
        grid_shape = (len(self.bin_coords), int_res)
        grid_coords = [
            self.bin_coords,
            np.linspace(-int_grid_xmax, +int_grid_xmax, int_res),
        ]
        grid_points = get_grid_points(grid_coords)
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

