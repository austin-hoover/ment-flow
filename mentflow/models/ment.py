"""Iterative maximum-entropy tomography (MENT) solvers.

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


def interpolate_1d(
    points: torch.Tensor,
    values: torch.Tensor, 
    int_points: torch.Tensor,
    method="linear"
) -> torch.Tensor:
    """Interpolate one-dimensional data."""
    points = grab(points)
    values = grab(values)
    int_points = grab(int_points)
    int_values = np.zeros(int_points.shape[0])
    if method == "nearest":
        fint = scipy.interpolate.interp1d(points, values, kind='nearest', bounds_error=False, fill_value=0.0)
        int_values = fint(int_points)
    elif method == "linear":
        fint = scipy.interpolate.interp1d(points, values, kind='linear', bounds_error=False, fill_value=0.0)
        int_values = fint(int_points)
    elif method == "pchip":
        interpolator = scipy.interpolate.PchipInterpolator(points, values, extrapolate=True)
        int_values = interpolator(int_points)
    else:
        raise ValueError("Invalid `interpolate` method.")
    int_values = torch.from_numpy(int_values)
    return int_values
    

def interpolate_dd(
    points: torch.Tensor,
    values: torch.Tensor, 
    int_points: torch.Tensor,
    method="linear",
) -> torch.Tensor:
    """Interpolate d-dimensional data."""
    if points.ndim == 1:    
        return interpolate_1d(points, values, int_points, method=method)

    points = grab(points)
    values = grab(values)
    int_points = grab(int_points)
    int_values = np.zeros(int_points.shape[0])

    int_values = scipy.interpolate.griddata(
        points,
        values,
        int_points,
        method=method,
        fill_value=0.0,
    )
    int_values = torch.from_numpy(int_values)
    return int_values


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


class MENT:
    """MENT reconstruction from projections.
    
    Parameters
    ----------
    d : int
        Dimension of the phase space.
    d_proj : int
        Dimension of the projected space. d_proj < d.
    transforms : list[callable]
        `transforms[i](x)` maps the input d-dimensional coordinates x to the ith
        measurement location.
    diagnostic : callable
        `diagnostic(x)` generates the measurement data. The entire forward process
        is `y_i = diagnostic(transforms[i](x))`. Must have the following parameters:
            - `bin_edges`: list[tensor], length d_proj; histogram bin edges
            - `bin_coords`: list[tensor], length d_proj; histogram bin centers
            - `axis`: int or tuple[int], length d_proj; the projection axis
    measurements : list[tensor]
        A list of measured m-dimensional projections. They should be normalized.
    prior : object
        A d-dimensional prior distribution. Must implement `prob(x)`. Defaults to uniform
        distribution with wide support [-100, 100].
    sampler : callable (optional)
        Implements `sample(f, n)` method to sample points from distribution function f.
        This is only used in the `simulate_sample` method.
    lagrange_functions : list[tensor]
        Lagrange functions ("h functions", "component functions") {h_i(u_i)} evaluated
        at each point on the the measurement axes u_i. We can treat them as continuous
        functions by interpolating between the points.
    interpolate : bool, str
        Interpolation method when evaluating the lagrange functions.
            - "nearest": nearest neighbor (scipy.interpolate.NearestNDInterpolator).
            - "linear": linear interpolation (scipy.interpolate.LinearNDInterpolator).
            - "pchip": 1D monotonic cubic splines (scipy.interpolate.PchipInterpolator).
    """
    def __init__(
        self,
        d: int,
        d_proj: int,
        transforms: List[Callable],
        diagnostic: Callable,
        measurements: List[torch.Tensor],
        prior: Any = None,
        interpolate: str = "linear",
        sampler: Optional[Callable] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        """Constructor."""
        self.device = device
        if self.device is None:
            self.device = torch.device("cpu")

        self.d = d
        self.d_proj = d_proj
        self.iteration = 0
        self.interpolate = interpolate
        self.transforms = transforms
        self.diagnostic = diagnostic
        self.measurements = self.set_measurements(measurements)

        self.prior = prior
        if self.prior is None:
            self.prior = UniformPrior(d=d, scale=100.0)

        self.lagrange_functions = self.initialize_lagrange_functions()
        self.sampler = sampler
        
    def _send(self, x):
        return x.type(torch.float32).to(self.device)

    def set_diagnostic(self, diagnostic: Callable):
        """Set the diagnostic (histogrammer)."""
        self.diagnostic = diagnostic
        if self.diagnostic is not None:
            self.proj_axis = self.diagnostic.axis
        return self.diagnostic

    def set_measurements(self, measurements: List[torch.Tensor]):
        """Set the measurement list."""
        self.measurements = measurements
        if self.measurements is None:
            self.measurements = []
        self.n_meas = len(self.measurements)
        return self.measurements

    def initialize_lagrange_functions(self) -> None:
        """Initialize the model to the prior distribution.

        Also set multipliers to zero where the measurement is zero. This will force
        zero density in regions outside the boundary defined by the backprojected
        measurements. (See Ref. [1].)
        """
        self.lagrange_functions = []
        for measurement in self.measurements:
            self.lagrange_functions.append((measurement > 0.0).float())
        return self.lagrange_functions

    def evaluate_lagrange_function(self, index: int, u: torch.Tensor) -> torch.Tensor:
        """Evaluate lagrange function h_i(u_i) at transformed point u."""
        points = self.diagnostic.bin_coords
        values = self.lagrange_functions[index]
        values = self._send(values)
        int_points = u[:, self.diagnostic.axis]
        int_values = interpolate_dd(points, values, int_points, method=self.interpolate)
        int_values = torch.clamp(int_values, 0.0, None)
        int_values = self._send(int_values)
        return int_values

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """Return the logarithm of the probability density at x."""
        # I think this should be general (product of h functions), but we'll need to check.
        log_prob = torch.ones(len(x))
        log_prob = self._send(log_prob)        
        for i, transform in enumerate(self.transforms):
            u = transform(x)
            h_func = self.evaluate_lagrange_function(i, u)
            log_prob += torch.log(h_func + 1.00e-12)
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
        """Normalize the projection."""
        # Should be straightforward to write general method.
        raise NotImplementedError

    def _simulate_integrate(self, index: int, **kws) -> torch.Tensor:
        """Compute the ith projection using numerical integration."""
        raise NotImplementedError

    def _simulate_sample(self, index: int, n: int = 100000, **kws) -> torch.Tensor:
        """Compute the ith projection using particle tracking + density estimation."""
        raise NotImplementedError

    def _simulate(self, index: int, method: str = "integrate", **kws) -> torch.Tensor:
        """Compute the ith projection.

        Parameters
        ----------
        index : int
            The measurement index.
        method : {"integrate", "sample"}
            Method to compute the projections of the distribution.
            - "integrate":
                Compute the density on a regular grid in the transformed space; evaluate
                the density on the grid using Jacobian determinant of transfer map.
            - "sample":
                Sample particles from the distribution, track the particles using the
                specified transform function, and estimate the projected density 
                using `self.diagnostic`.
        **kws
            Key word arguments passed to `self._simulate_integrate` (if method="integrate")
            or `self._simulate_sample` (if method="sample").

        Returns
        -------
        tensor, shape (len(self.diagnostic.bin_coords),)
            The projection onto the ith measurement axis.
        """
        _functions = {
            "integrate": self._simulate_integrate,
            "sample": self._simulate_sample,
        }
        _function = _functions[method]
        return _function(index=index, **kws)

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
        raise NotImplementedError

    def parameters(self):
        return

    def save(path):
        return

    def load(path):
        return


# I'm guessing that higher dimensional versions of MENT will follow the same pattern, so 
# we wont' need to write separate classes for each case.
        

class MENT_2D1D(MENT):
    """2D MENT reconstruction from 1D projections."""
    def __init__(self, **kws) -> None:
        super().__init__(d=2, d_proj=1, **kws)

        # Use grid sampler by default.
        if self.sampler is None:
            xmax = 1.5 * max(self.diagnostic.bin_edges)
            limits = 2 * [(-xmax, +xmax)]
            self.sampler = GridSampler(limits=limits, res=200)

    def _normalize_projection(self, projection: np.ndarray) -> np.ndarray:
        """Normalize one-dimensional projection."""
        bin_volume = self.diagnostic.bin_edges[1] - self.diagnostic.bin_edges[0]
        normalization = projection.sum() * bin_volume
        return projection / normalization

    def _simulate_integrate(self, index: int, xmax: float = None, res: int = 150, **kws) -> torch.Tensor:
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
        tensor, shape (len(self.diagnostic.bin_coords),)
            The projection onto the ith measurement axis.
        """
        # Compute the density on a regular grid in the transformed space.
        # We do this by flowing backward and tracking the Jacobian determinant
        # (determinant = 1 for linear transformations).
        if xmax is None:
            xmax = 1.5 * max(self.diagnostic.bin_coords)
        int_grid_xmax = xmax  # integration limits
        int_res = res  # integration grid resolution
        grid_shape = (len(self.diagnostic.bin_coords), int_res)
        grid_coords = [
            self.diagnostic.bin_coords,  # measurement axis
            torch.linspace(-int_grid_xmax, +int_grid_xmax, int_res),  # integration axis
        ]
        grid_coords = [self._send(c) for c in grid_coords]
        grid_points = get_grid_points_torch(grid_coords)
        grid_points = self._send(grid_points)
        transform = self.transforms[index]
        grid_points_in = transform.inverse(grid_points)
        prob_in = torch.exp(self.log_prob(grid_points_in))

        # Assume det(J) = 1 for now; in the future, we should add a `forward_and_log_det`
        # method to `transform` object to track the Jacobian determinant.
        prob_out = prob_in

        # Compute the projection in the transformed space.
        prob_out = prob_out.reshape(grid_shape)
        prediction = torch.sum(prob_out, dim=1)
        prediction = self._normalize_projection(prediction)
        return prediction

    def _simulate_sample(self, index: int, n: int = 100000, **kws) -> torch.Tensor:
        """Compute the ith projection using particle tracking + density estimation.

        This function samples particles from the model distribution, tracks the
        particles using the ith transform function, and estimates the projected
        density using a histogram.

        Consequently, the forward process does not need to be invertible. 

        Parameters
        ----------
        index : int
            The measurement index.
        n : int
            The number of particles to sample.

        Returns
        -------
        tensor, shape (len(self.diagnostic.bin_coords),)
            The projection onto the ith measurement axis.
        """
        transform = self.transforms[index]
        x = self.sample(n)
        x = self._send(x)
        u = transform(x)
        prediction = self.diagnostic(u)
        prediction = self._normalize_projection(prediction)
        return prediction

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
            for j in range(len(self.lagrange_functions[i])):
                g_meas = float(measurement[j])
                g_pred = float(prediction[j])
                if (g_meas != 0.0) and (g_pred != 0.0):
                    self.lagrange_functions[i][j] *= (g_meas / g_pred)
        self.iteration += 1


