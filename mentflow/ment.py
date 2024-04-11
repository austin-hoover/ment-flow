"""Iterative maximum-entropy tomography (MENT) solver.

References
----------
[1] https://doi.org/10.1109/78.80970
[2] https://doi.org/10.1103/PhysRevAccelBeams.25.042801
"""
import math
from typing import Any
from typing import Callable
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
import scipy.interpolate
import torch

from mentflow.sim import forward
from mentflow.utils import coords_from_edges
from mentflow.utils import get_grid_points
from mentflow.utils import grab
from mentflow.utils import unravel


class LagrangeFunction:
    def __init__(self, coords: List[torch.Tensor], values: torch.Tensor, interpolation_kws: dict = None) -> None:
        self.coords = coords
        self.values = values
        
        self.interpolator = None
        self.interpolation_kws = interpolator_kws
        if self.interpolation_kws is None:
            self.interpolation_kws = dict()
        self.interpolation_kws.setdefault("method", "linear")
        self.interpolation_kws.setdefault("bounds_error", False)
        self.interpolation_kws.setdefault("fill_value", 0.0)
        
        self.set_values(values)

    def set_values(self, values: torch.Tensor) -> None:
        self.values = values

        _values = grab(values)
        if _values.ndim == 1:
            _coords = [grab(self.coords)]
        else:
            _coords = [grab(c) for c in self.coords]
                    
        self.interpolator = scipy.interpolate.RegularGridInterpolator(
            _coords,
            _values,
            **interpolation_kws
        )

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return torch.from_numpy(self.interpolator(x))
    

class MENT:
    """Iterative maximum-entropy tomography (MENT) solver.

    Attributes
    ----------
    lagrange_functions : list[LagrangeFunction]
        Lagrange functions ("h functions", "component functions") evaluated on the
        measurement grids. We treat them as continuous functions by interpolating
        between grid points.
    epoch : int
        The current epoch (MENT iteration looping over all measurements).

    References
    ----------
    [1] https://doi.org/10.1109/78.80970
    [2] https://doi.org/10.1103/PhysRevAccelBeams.25.042801
    """
    def __init__(
        self,
        ndim: int,
        transforms: List[Callable],
        diagnostics: List[List[Callable]],
        measurements: List[List[torch.Tensor]],
        discrepancy_function: Callable,
        prior: Any = None,
        interpolation: dict = None,
        mode: str = "integrate",
        integration_grid_limits: List[Tuple[float]] = None,
        integration_grid_shape: Tuple[int] = None,
        sampler: Optional[Callable] = None,
        n_samples: int = 1000000,
        device: Optional[torch.device] = None,
        verbose: bool = False,
    ) -> None:
        """Constructor.
        
        Parameters
        ----------
        ndim : int
            Dimension of reconstructed phase space.
        transforms : list[nn.Module]
            Mappings between reconstruction and measurement locations.
        diagnostics : list[list[torch.nn.Module]
            `diagnostics[i]` is a list of diagnostics applied after the ith transform.
            - Must implement `diagnostic(x)`, producing measurement data.
            - Must have the following parameters:
                - `bin_edges`: tensor or list[tensor]; histogram bin edges
                - `axis`: int or tuple[int]; the projection axis
                - `ndim`: int; the projected dimension
        measurements : list[list[tensor]
            Measured data corresponding to each diagnostic.
        discrepancy_function: Callable
            Computes scalar difference between measurements and predictions.
            Call signature: `discrepancy_function(pred: torch.Tensor, meas: torch.Tensor) -> torch.Tensor (float)`.
        prior : Any
            Prior distribution implementing `prior.prob(x: torch.Tensor) -> torch.Tensor`. 
            Defaults to a uniform distribution with wide suppport ([-100, 100]).
        interpolation_kws : dict
            Key word arguments passed to scipy.interpolate.RegularGridInterpolator.
        mode: {"integrate", "sample"}
            Whether to use numerical integration or sampling + particle tracking to compute projections.
        integration_grid_limits : list[list[ list[tuple[float] ]]] (optional)
            The integration grid limits [(xmin, xmax), (ymin, ymax), ...] for each measurement.
        integration_grid_shape : list[list[tuple[int]]] (optional)
            The integration grid shape (n1, n2, ...) for each measurement.
        sampler : callable (optional)
            Method to sample particles from probability density function.
            Call signature: `sample(prob_func: Callable, size: int) -> torch.Tensor`,
            where `prob_func(x: torch.Tensor) -> torch.Tensor returns the probability density
            at point `x`.
        n_samples : int (optional)
            The number of samples to draw if mode="sample".
        device : str, torch.device, None
            A torch device if running on GPU. This is not working currently; run on cpu.
        """
        self.device = device
        self.verbose = verbose
        self.mode = mode
        self.ndim = ndim
        self.epoch = 0

        self.d_meas = [[]]
        self.d_int = [[]]

        self.transforms = transforms
        self.diagnostics = self.set_diagnostics(diagnostics)
        self.measurements = self.set_measurements(measurements)
        self.discrepancy_function = discrepancy_function

        self.prior = prior
        if self.prior is None:
            self.prior = UniformPrior(d=d, scale=100.0)

        self.integration_grid_limits = integration_grid_limits
        self.integration_grid_shape = integration_grid_shape

        self.sampler = sampler
        self.n_samples = n_samples

        self.lagrange_functions = self.initialize_lagrange_functions(**interpolation_kws)

    def send(self, x):
        return x.type(torch.float32).to(self.device)

    def set_diagnostics(self, diagnostics: List[List[Callable]]):
        self.diagnostics = diagnostics
        if self.diagnostics is None:
            self.diagnostics = [[]]
        return self.diagnostics

    def set_measurements(self, measurements: List[List[torch.Tensor]]):
        self.measurements = measurements
        if self.measurements is None:
            self.measurements = [[]]
        return self.measurements

    def initialize_lagrange_functions(self, **interp_kws) -> List[List[torch.Tensor]]:
        self.lagrange_functions = []
        for index in range(len(self.measurements)):
            self.lagrange_functions.append([])
            for measurement, diagnostic in zip(self.measurements[index], self.diagnostics[index]):
                edges = diagnostic.bin_edges
                if measurement.ndim == 1:
                    coords = coords_from_edges(edges)
                else:
                    coords = [coords_from_edges(e) for e in edges]
                values = (meas > 0.0).float()
                lagrange_function = LagrangeFunction(coords, values, **interp_kws)
                self.lagrange_functions[-1].append(lagrange_function)
        return self.lagrange_functions

    def normalize_projection(self, projection: torch.Tensor, index: int, diag_index: int) -> torch.Tensor:
        diagnostic = self.diagnostics[index][diag_index]
        bin_volume = 1.0
        if diagnostic.ndim == 1:
            bin_volume = diagnostic.bin_edges[1] - diagnostic.bin_edges[0]
        else:
            bin_volume = math.prod((e[1] - e[0]) for e in diagnostic.bin_edges)
        return projection / projection.sum() / bin_volume

    def get_meas_points(self, index: int, diag_index: int) -> torch.Tensor:
        """Return measurement grid points."""
        diagnostic = self.diagnostics[index][diag_index]
        if self.d_meas[index][diag_index] == 1:
            return coords_from_edges(diagnostic.bin_edges)
        else:
            coords = [coords_from_edges(e) for e in diagnostic.bin_edges]
            return get_grid_points(*coords)

    def get_integration_points(self, index: int, diag_index: int, method: str = "grid") -> torch.Tensor:
        """Return integration points in transformed space."""
        limits = self.integration_grid_limits[index][diag_index]
        shape = self.integration_grid_shape[index][diag_index]
        diagnostic = self.diagnostics[index][diag_index]
        meas_axis = diagnostic.axis
        if type(meas_axis) is int:
            meas_axis = (meas_axis,)
        int_axis = tuple([axis for axis in range(self.d) if axis not in meas_axis])

        if method == "grid":
            int_coords = [
                torch.linspace(limits[k][0], limits[k][1], shape[k]) for k in range(len(int_axis))
            ]
            int_coords = [self.send(c) for c in int_coords]

            int_points = None
            if len(int_axis) == 1:
                int_coords = int_coords[0]
                int_points = int_coords
            else:
                int_points = get_grid_points(*int_coords)
        elif method == "uniform":
            raise NotImplementedError
        elif method == "gaussian":
            raise NotImplementedError
        else:
            raise ValueError("Invalid integration method.")

        int_points = self.send(int_points)
        return int_points

    def evaluate_lagrange_function(self, u: torch.Tensor, index: int, diag_index: int) -> torch.Tensor:
        """Evaluate lagrange function at transformed coordinates u."""
        diagnostic = self.diagnostics[index][diag_index]
        lagrange_function = self.lagrange_functions[index][diag_index]
        u_proj = diagnostic.project(u)
        u_proj = grab(u_proj)
        values = lagrange_function(u_proj)
        values = self.send(values)
        return values

    def log_prob(self, x: torch.Tensor, pad: float = 1.00e-12) -> torch.Tensor:
        """Return the log probability density at points x."""
        return torch.log(self.prob(x) + pad)

    def prob(self, x: torch.Tensor) -> torch.Tensor:
        """Return the probability density at points x."""
        prob = torch.ones(x.shape[0])
        prob = self.send(prob)
        for index, transform in enumerate(self.transforms):
            u = transform(x)
            for diag_index, diagnostic in enumerate(self.diagnostics[index]):
                h = self.evaluate_lagrange_function(u, index, diag_index)
                h = torch.clamp(h, 0.0, 1.00e+10)  # stability
                prob *= h
        prob *= torch.exp(self.prior.log_prob(x))
        return prob

    def sample(self, n: int) -> torch.Tensor:
        """Sample n particles from the distribution."""
        x = self.sampler(self.prob, n=n)
        x = self.send(x)
        return x

    def sample_and_log_prob(self, n: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample n particles from the distribution and evaluate log probability at each point."""
        x = self.sample(n)
        return x, self.log_prob(x)

    def discrepancy_vector(self, predictions: List[List[torch.Tensor]]) -> List[torch.Tensor]:
        """Convenience function to compute simulation-measurement discrepancy."""
        discrepancy_vector = []
        for pred, meas in zip(unravel(predictions), unravel(self.measurements)):
            discrepancy = self.discrepancy_function(pred, meas)
            discrepancy_vector.append(discrepancy)
        return discrepancy_vector

    def _simulate_integrate(self, index: int, diag_index: int, int_method: str = "grid") -> torch.Tensor:
        """Simulate projection by numerically integrating distribution function."""
        limits = self.integration_grid_limits[index][diag_index]
        shape = self.integration_grid_shape[index][diag_index]
        diagnostic = self.diagnostics[index][diag_index]
        transform = self.transforms[index]

        # Define measurement axis.
        meas_axis = diagnostic.axis
        if type(meas_axis) is int:
            meas_axis = (meas_axis,)

        # Define integration axis.
        int_axis = tuple([axis for axis in range(self.d) if axis not in meas_axis])

        # Get measurement and integration points.
        meas_points = self.get_meas_points(index, diag_index)
        meas_points = self.send(meas_points)
        int_points = self.get_integration_points(index, diag_index, int_method)
        int_points = self.send(int_points)

        # Initialize transformed coordinate array.
        u = torch.zeros((int_points.shape[0], self.d))
        u = self.send(u)
        for k, axis in enumerate(int_axis):
            if len(int_axis) == 1:
                u[:, axis] = int_points
            else:
                u[:, axis] = int_points[:, k]

        # Compute integral.
        prediction = torch.zeros(meas_points.shape[0])
        for i, meas_point in enumerate(meas_points):
            # Update the coordinates in the measurement plane.
            for k, axis in enumerate(meas_axis):
                if self.d_meas[index][diag_index] == 1:
                    u[:, axis] = meas_point
                else:
                    u[:, axis] = meas_point[k]
            # Compute the probability density at the integration points.
            x = transform.inverse(u)
            prob = self.prob(x)  # assume symplectic transform
            # Integrate (ignore scaling factor)
            prediction[i] = torch.sum(prob)

        # Reshape the flattened projection.
        if self.d_meas[index][diag_index] > 1:
            prediction = prediction.reshape(diagnostic.shape)

        # Normalize the projection.
        prediction = self.send(prediction)
        prediction = self.normalize_projection(prediction, index, diag_index)
        return prediction

    def _simulate_sample(self, index: int, diag_index: int) -> torch.Tensor:
        """Simulate projection by sampling and tracking particles."""
        x = self.sample(int(self.n_samples))
        x = self.send(x)
        transform = self.transforms[index]
        diagnostic = self.diagnostics[index][diag_index]
        prediction = diagnostic(transform(x))
        prediction = self.normalize_projection(prediction, index, diag_index)
        return prediction

    def _simulate(self, index: int, diag_index: int, **kws) -> torch.Tensor:
        """Simulate projection."""
        _functions = {
            "integrate": self._simulate_integrate,
            "sample": self._simulate_sample,
        }
        _function = _functions[self.mode]
        return _function(index, diag_index, **kws)

    def gauss_seidel_iterate(self, omega: float = 1.0, thresh: float = 1.0e-10, **kws) -> None:
        """Perform Gauss-Seidel iteration to update Lagrange functions.

        Parameters
        ----------
        omega : float
            Learning rate in range [0.0, 1.0].
            h -> h * (1 + omega * ((g / g*) - 1))
        thresh: float
            Fractional threshold applied to simulated projections (g*).
        **kws
            Key word arguments passed to `self._simulate`.
        """
        for index, transform in enumerate(self.transforms):
            if self.verbose:
                print(f"index={index}")
            for diag_index, diagnostic in enumerate(self.diagnostics[index]):
                lagrange_function = self.lagrange_functions[index][diag_index]
                measurement = self.measurements[index][diag_index]
                
                prediction = self._simulate(index, diag_index, **kws)
                thresh = torch.max(prediction) * thresh
                prediction[prediction < thresh] = 0.0

                # print(index, diag_index, lagrange_function.values.max())
                
                shape = lagrange_function.shape
                lagrange_function.values = torch.ravel(lagrange_function.values)
                for k, (g_meas, g_pred) in enumerate(zip(torch.ravel(measurement), torch.ravel(prediction))):
                    if (g_meas != 0.0) and (g_pred != 0.0):
                        frac = g_meas
                        lagrange_function.values[k] *= 1.0 + omega * ((g_meas / g_pred) - 1.0)
                lagrange_function.values = torch.reshape(lagrange_function.values, shape)
                lagrange_function.set_values(lagrange_function.values)
                self.lagrange_functions[index][diag_index] = lagrange_function
        self.epoch += 1

    def simulate(self, **kws):
        """Simulate all projections."""
        predictions = []
        if self.mode == "integrate":
            for index, transform in enumerate(self.transforms):
                predictions.append([])
                for diag_index, diagnostic in enumerate(self.diagnostics[index]):
                    prediction = self._simulate_integrate(index, diag_index, **kws)
                    predictions[-1].append(prediction)
        elif self.mode == "sample":
            n = kws.get("n", 100000)
            x = self.sample(n)
            x = self.send(x)
            predictions = forward(x, self.transforms, self.diagnostics)
        return predictions

    def parameters(self):
        return self.lagrange_functions

    def save(self, path) -> None:
        state = {
            "lagrange_functions": self.lagrange_functions,
            "epoch": self.epoch,
            "transforms": self.transforms,
            "diagnostics": self.diagnostics,
            "measurements": self.measurements,
            "prior": self.prior,
            "d": self.d,
            "sampler": self.sampler,
        }
        torch.save(state, path)

    def load(self, path, device=None):
        state = torch.load(path, map_location=device)

        self.lagrange_functions = state["lagrange_functions"]
        self.epoch = state["epoch"]
        self.transforms = state["transforms"]
        self.diagnostics = state["diagnostics"]
        self.measurements = state["measurements"]
        self.prior = state["prior"]
        self.d = state["d"]
        self.sampler = state["sampler"]

        self.to(device)

    def to(self, device):
        self.device = device
        if self.transforms is not None:
            for i in range(len(self.transforms)):
                self.transforms[i] = self.transforms[i].to(device)
        if self.diagnostics is not None:
            for i in range(len(self.diagnostics)):
                for j in range(len(self.diagnostics[i])):
                    self.diagnostics[i][j] = self.diagnostics[i][j].to(device)
        if self.measurements is not None:
            for i in range(len(self.measurements)):
                for j in range(len(self.measurements[i])):
                    self.measurements[i][j] = self.measurements[i][j].to(device)
        if self.sampler is not None:
            self.sampler = self.sampler.to(device)
        if self.prior is not None:
            self.prior = self.prior.to(device)
        for i in range(len(self.lagrange_functions)):
            for j in range(len(self.lagrange_functions[i])):
                self.lagrange_functions[i][j].values = self.send(self.lagrange_functions[i][j].values)
        return self
