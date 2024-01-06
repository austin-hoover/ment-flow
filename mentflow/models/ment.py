"""Iterative maximum-entropy tomography (MENT) solvers.

References
----------
[1] https://doi.org/10.1109/78.80970
[2] https://doi.org/10.1103/PhysRevAccelBeams.25.042801
"""
import abc
import math
import time
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

from mentflow.loss import get_loss_function
from mentflow.sample import GridSampler
from mentflow.utils import get_grid_points_torch
from mentflow.utils import grab
from mentflow.utils import unravel


def interpolate_1d(
    points: torch.Tensor, 
    values: torch.Tensor, 
    int_points: torch.Tensor, 
    method="linear",
) -> torch.Tensor:
    """Interpolate one-dimensional data."""
    points = grab(points)
    values = grab(values)
    int_points = grab(int_points)
    int_values = np.zeros(int_points.shape[0])
    if method == "nearest":
        fint = scipy.interpolate.interp1d(points, values, kind="nearest", bounds_error=False, fill_value=0.0)
        int_values = fint(int_points)
    elif method == "linear":
        fint = scipy.interpolate.interp1d(points, values, kind="linear", bounds_error=False, fill_value=0.0)
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

        cov = torch.eye(d) * (scale**2)
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
    """MENT reconstruction model.

    This should work for any dimension d and projected dimension d', but it has only
    been tested for d = 2, d' = 1.

    Parameters
    ----------
    d : int
        Dimension of the reconstructed phase space.
    transforms : list[callable]
        `transforms[i](x)` maps the input d-dimensional coordinates x to the ith
        measurement location.
    diagnostics : list[callable]
        Histogram diagnostics to apply after each transform. 
        Must implement `diagnostic(x)`, producting measurement data. 
        Must have the following parameters:
            - `bin_edges`: tensor or list[tensor]; histogram bin edges
            - `bin_coords`: tensor or list[tensor]; histogram bin centers
            - `axis`: int or tuple[int]; the projection axis
    measurements : list[list[tensor], shape (len(diagnostics))], shape (len(transforms))]
        The measured projections. They should be normalized. 
    prior : object
        A d-dimensional prior distribution. Must implement `prob(x)`. Defaults to a
        uniform distribution with wide support [-100, 100].
    sampler : callable (optional)
        Implements `sample(f, n)` method to sample points from distribution function f.
        This is only used in the `simulate_sample` method.
    interpolate : bool, str
        Interpolation method when evaluating the lagrange functions.
            - "nearest": nearest neighbor (scipy.interpolate.NearestNDInterpolator).
            - "linear": linear interpolation (scipy.interpolate.LinearNDInterpolator).
            - "pchip": 1D monotonic cubic splines (scipy.interpolate.PchipInterpolator).

    Attributes
    ----------
    lagrange_functions : list[tensor]
        Lagrange functions ("h functions", "component functions") {h_i(y_i)} evaluated
        at each point on the the measurement axes y_i. We can treat them as continuous
        functions by interpolating between the points.
    d_meas : int
        Dimension of measurement axis. (d_meas < d)
    d_int : int
        Dimension of integration axis. (d_int = d - d_meas)
    """
    def __init__(
        self,
        d: int,
        transforms: List[Callable],
        diagnostics: List[Callable],
        measurements: List[List[torch.Tensor]],
        discrepancy_function: str = "kld",
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
        self.d_meas = self.d_int = None
        self.iteration = 0
        self.interpolate = interpolate

        self.transforms = transforms
        self.diagnostics = self.set_diagnostics(diagnostics)
        self.measurements = self.set_measurements(measurements)
        self.discrepancy_function = get_loss_function(discrepancy_function)

        self.prior = prior
        if self.prior is None:
            self.prior = UniformPrior(d=d, scale=100.0)

        self.lagrange_functions = self.initialize_lagrange_functions()
        self.sampler = sampler

    def send(self, x):
        """Send tensor to torch device."""
        return x.type(torch.float32).to(self.device)

    def set_diagnostics(self, diagnostics: List[Callable]):
        """Set the diagnostics."""
        self.diagnostics = diagnostics
        if self.diagnostics is not None:
            self.meas_axis = []
            self.meas_coords = []
            self.meas_points = []
            for diagnostic in diagnostics:
                axis = diagnostic.axis
                if type(axis) is int:
                    self.meas_coords.append(diagnostic.bin_coords)
                    self.meas_points.append(diagnostic.bin_coords)
                else:
                    coords = [c for c in diagnostic.bin_coords]
                    self.meas_coords.append(coords)
                    self.meas_points = self.send(get_grid_points_torch(coords))
                self.meas_axis.append(axis)
        return self.diagnostics

    def set_measurements(self, measurements: List[List[torch.Tensor]]):
        """Set the measurement list."""
        self.measurements = measurements
        if self.measurements is None:
            self.measurements = [[]]
        self.n_meas = len(list(unravel(self.measurements)))
        self.d_meas, self.d_int = [], []
        if self.n_meas > 0:
            for measurement in self.measurements[0]:
                self.d_meas.append(measurement.ndim)
                self.d_int.append(self.d - measurement.ndim)
        return self.measurements

    def _normalize_projection(self, projection: torch.Tensor, diag_index=0) -> torch.Tensor:
        """Normalize the projection."""
        j = diag_index
        diagnostic = self.diagnostics[j]
        if self.d_meas[j] == 1:
            bin_volume = diagnostic.bin_edges[1] - diagnostic.bin_edges[0]
        else:
            bin_volume = math.prod((e[1] - e[0]) for e in diagnostic.bin_edges)
        return projection / projection.sum() / bin_volume

    def initialize_lagrange_functions(self) -> None:
        """Initialize the model to the prior distribution.

        Also set multipliers to zero where the measurement is zero. This will force zero 
        density in regions outside the boundary defined by the backprojected measurements.
        See Ref. [1].
        """
        self.lagrange_functions = []
        for index in range(len(self.measurements)):
            self.lagrange_functions.append([])
            for measurement in self.measurements[index]:
                self.lagrange_functions[-1].append((measurement > 0.0).float())
        return self.lagrange_functions

    def evaluate_lagrange_function(
        self, 
        u: torch.Tensor,
        index: int, 
        diag_index: int = 0, 
    ) -> torch.Tensor:
        """Evaluate lagrange function h(u) at transformed points u."""
        points = self.meas_points[diag_index]
        values = self.lagrange_functions[index][diag_index].ravel()
        values = self.send(values)
        axis = self.meas_axis[diag_index]
        int_points = u[:, axis]
        int_values = interpolate_dd(points, values, int_points, method=self.interpolate)
        int_values = torch.clamp(int_values, 0.0, None)
        int_values = self.send(int_values)
        return int_values

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """Return the logarithm of the probability density at x."""
        log_prob = torch.ones(len(x))
        log_prob = self.send(log_prob)
        for index, transform in enumerate(self.transforms):
            u = transform(x)
            for diag_index, diagnostic in enumerate(self.diagnostics):
                h = self.evaluate_lagrange_function(u, index, diag_index)
                log_prob += torch.log(h + 1.00e-12)
        log_prob += self.prior.log_prob(x)
        return log_prob

    def prob(self, x: torch.Tensor) -> torch.Tensor:
        """Return the probability density at x."""
        return torch.exp(self.log_prob(x))

    def sample(self, n: int) -> torch.Tensor:
        """Sample particles from the distribution."""
        x = self.sampler(log_prob_func=self.log_prob, n=n)
        x = self.send(x)
        return x

    def sample_and_log_prob(self, n: int) -> Tuple[np.ndarray, np.ndarray]:
        """Sample particles x and compute log(prob(x))."""
        x = self.sample(n)
        log_prob = self.log_prob(x)
        return (x, log_prob)

    def discrepancy(self, predictions) -> List[torch.Tensor]:
        """Compute simulation-measurement discrepancy vector."""
        discrepancy_vector = []
        for prediction, measurement in zip(unravel(predictions), unravel(self.measurements)):
            discrepancy = self.discrepancy_function(prediction, measurement)
            discrepancy_vector.append(discrepancy)
        return discrepancy_vector

    def _simulate_integrate(
        self, 
        index: int,
        diag_index: int,
        limits: List[Tuple[float]],
        shape: Tuple[int], 
        **kws
    ) -> torch.Tensor:
        """Simulate projection by numerically integrating the distribution function.

        Parameters
        ----------
        index : int
            The transform index.
        diag_index : int
            The diagnostic index.
        limits : list[tuple[float]]
            List of (min, max) coordinate along each axis of integration grid.
        shape : tuple[int]
            Resolution along each axis of integration grid.

        Returns
        -------
        tensor
            The projection onto the measurement axis.
        """                
        # Define the measurement axis.
        diagnostic = self.diagnostics[diag_index]
        meas_axis = self.meas_axis[diag_index]
        if self.d_meas[diag_index] == 1:
            meas_axis = (meas_axis,)
        d_meas = self.d_meas[diag_index]

        # Get the measurement points.
        meas_points = self.meas_points[diag_index]

        # Define the integration axis.
        int_axis = [axis for axis in range(self.d) if axis not in meas_axis]

        # Get the integration points in the transformed space.
        int_coords = [
            torch.linspace(limits[k][0], limits[k][1], shape[k]) 
            for k in range(len(int_axis))
        ]
        int_coords = [self.send(c) for c in int_coords]
        if len(int_axis) == 1:
            int_coords = int_coords[0]
            int_points = int_coords
        else:
            int_points = get_grid_points_torch(int_coords)
        int_points = self.send(int_points)

        # Initialize transformed coordinate vector u.
        u = torch.zeros((int_points.shape[0], self.d))
        for k, axis in enumerate(int_axis):
            if len(int_axis) == 1:
                u[:, axis] = int_points
            else:
                u[:, axis] = int_points[:, k]

        # Integrate
        prediction = torch.zeros(meas_points.shape[0])
        transform = self.transforms[index]
        for i, meas_point in enumerate(meas_points):
            # Append the measurement point to the integration points.
            for k, axis in enumerate(meas_axis):
                if d_meas == 1:
                    u[:, axis] = meas_point
                else:
                    u[:, axis] = meas_point[k]
            # Compute the probability density at the integration points.
            x, ladj = transform.inverse_and_ladj(u)
            log_prob = self.log_prob(x) + ladj  # check sign
            prob = torch.exp(log_prob)
            # Sum over the integration points.
            prediction[i] = torch.sum(prob)
            
        # Reshape and normalize the projection.
        if d_meas > 1:
            prediction = prediction.reshape(diagnostic.shape)
        prediction = self._normalize_projection(prediction, diag_index)
        return prediction

    def _simulate_sample(self, index: int, diag_index: int = 0, n: int = 100000, **kws) -> torch.Tensor:
        """Compute projection by sampling and tracking particles.

        Parameters
        ----------
        index : int
            The transform index.
        diag_index : int
            The diagnostic index.
        n : int
            The number of particles to sample.

        Returns
        -------
        list[tensor]
            The projections onto the measurement axis.
        """
        x = self.sample(n)
        x = self.send(x)
        transform = self.transforms[index]
        diagnostic = self.diag_index[diag_index]
        prediction = diagnostic(transform(x))
        prediction = self._normalize_projection(prediction, diag_index)
        return prediction

    def _simulate(self, index: int, diag_index: int = 0, method: str = "integrate", **kws) -> torch.Tensor:
        """Compute the ith projection.

        Parameters
        ----------
        index : int
            The transform index.
        diag_index : int
            The diagnostic index.
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
        return _function(index=index, diag_index=diag_index, **kws)
        
    def gauss_seidel_iterate(self, omega=1.0, sim_method="integrate", **sim_method_kws) -> None:
        """Execute Gauss-Seidel iteration to update lagrange functions.

        Parameters
        ----------
        omega : float
            Under-relaxation parameter in range (0, 1). 
            h -> h * (1 + omega * ((g / g*) - 1))
        sim_method : str
            Simulation method {"integrate", "sample"}.
        **kwargs
            Key word arguments passed to `self._simulate`.
        """
        for index in range(len(self.measurements)):
            for diag_index, measurement in enumerate(self.measurements[index]):
                prediction = self._simulate(index, diag_index, method=sim_method, **sim_method_kws)
                lagrange_function = self.lagrange_functions[index][diag_index]
                shape = lagrange_function.shape
                lagrange_function = lagrange_function.ravel()
                for k, (g_meas, g_pred) in enumerate(zip(measurement.ravel(), prediction.ravel())):                
                    if (g_meas != 0.0) and (g_pred != 0.0):
                        lagrange_function[k] *= (1.0 + omega * ((g_meas / g_pred) - 1.0))                        
                lagrange_function = lagrange_function.reshape(shape)
                self.lagrange_functions[index][diag_index] = lagrange_function
        self.iteration += 1

    def simulate(self, method="integrate", x=None, **kws):
        """Simulate all projections."""            
        predictions = []
        if method == "integrate":
            for index, transform in enumerate(self.transforms):
                predictions.append([])
                for diag_index, diagnostic in enumerate(self.diagnostics):
                    prediction = self._simulate_integrate(index, diag_index, **kws)
                    predictions[-1].append(prediction)
        elif method == "sample":
            if x is None:
                n = kws.get("n", 100000)
                x = self.sample(n)
                x = self.send(x)
            predictions = []
            for index, transform in enumerate(self.transforms):
                u = transform(x)
                predictions.append([])
                for diag_index, diagnostic in enumerate(self.diagnostics):
                    prediction = diagnostic(u)
                    prediction = self._normalize_projection(prediction, diag_index)
                    predictions[-1].append(prediction)
        return predictions

    def parameters(self):
        return self.lagrange_functions

    def save(self, path) -> None:
        state = {
            "lagrange_functions": self.lagrange_functions,
            "iteration": self.iteration,
            "transforms": self.transforms,
            "diagnostics": self.diagnostics,
            "measurements": self.measurements,
            "prior": self.prior,
            "d": self.d,
            "sampler": self.sampler,
        }
        torch.save(state, path)

    def load(self, path, device) -> None:
        state = torch.load(path, map_location=device)

        self.lagrange_functions = state["lagrange_functions"]
        self.iteration = state["iteration"]
        self.transforms = state["transforms"]
        self.diagnostics = state["diagnostics"]
        self.measurements = state["measurements"]
        self.prior = state["prior"]
        self.d = state["d"]
        self.sampler = state["sampler"]
        
        self.to(device)

    def to(self, device):
        if self.transforms is not None:
            for i in range(len(self.transforms)):
                self.transforms[i] = self.transforms[i].to(device)
        if self.diagnostics is not None:
            for i in range(len(self.diagnostics)):
                self.diagnostics[i] = self.diagnostics[i].to(device)
        if self.measurements is not None:
            for i in range(len(self.measurements)):
                self.measurements[i] = self.measurements[i].to(device)
        return self

