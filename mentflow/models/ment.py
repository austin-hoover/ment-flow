"""Iterative maximum-entropy tomography (MENT) solver.

References
----------
[1] https://doi.org/10.1109/78.80970
[2] https://doi.org/10.1103/PhysRevAccelBeams.25.042801
"""
import typing
from typing import Callable
from typing import Iterable
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
import scipy.interpolate

from mentflow.utils import get_grid_points
from mentflow.utils import sample_hist


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


class MENT_2D1D_numpy:
    """2D MENT reconstruction from 1D projections. [NumPy version]

    May replace with PyTorch version for easier comparisons with MENT-Flow.
    """
    def __init__(
        self,
        transforms: List[Callable],
        diagnostic: Callable,
        measurements: List[np.ndarray],
        prior=None,
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
        """
        self.d = 2
        self.iteration = 0
        self.transforms = transforms
        self.diagnostic = self.set_diagnostic(diagnostic)
        self.measurements = self.set_measurements(measurements)
        self.prior = prior
        if self.prior is None:
            self.prior = UniformPrior(scale=100.0)
        self.lagrange_multipliers = self.initialize_lagrange_multipliers()

    def set_diagnostic(self, diagnostic: Callable):
        self.diagnostic = diagnostic
        if self.diagnostic is not None:
            self.n_bins = self.diagnostic.bin_edges
            self.bin_edges = self.diagnostic.bin_edges
            self.bin_coords = 0.5 * (self.bin_edges[:-1] + self.bin_edges[1:])
        return self.diagnostic

    def set_measurements(self, measurements: List[np.ndarray]):
        self.measurements = measurements
        if self.measurements is None:
            self.measurements = []
        self.n_meas = self.n_constraints = len(self.measurements)
        return self.measurements

    def initialize_lagrange_multipliers(self):
        """Initialize model to prior."""
        self.lagrange_multipliers = []
        for i, measurement in enumerate(self.measurements):
            self.lagrange_multipliers.append([])
            for j in range(len(measurement)):
                mij = float(measurement[j] > 0)
                self.lagrange_multipliers[-1].append(mij)
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
        idx = list(range(len(self.measurements)))

        _prob = np.ones(len(x))
        for i in idx:
            x_out = self.transforms[i](x)
            _sum = np.zeros(len(x_out))
            for j in range(len(self.measurements[i])):
                _sum += self._chi(x_out, i, j) * self.lagrange_multipliers[i][j]
            _prob *= _sum
        return _prob * self.prior.prob(x)

    def gauss_seidel_iterate(
        self,
        method: str = "integrate",
        xmax: float = None,
        res: int = 150,
        n_samples: int = 1000000,
    ) -> None:
        """Execute Gauss-Seidel iteration to update lagrange multipliers."""
        if xmax is None:
            xmax = 1.25 * np.max(self.bin_edges)

        for i in range(len(self.lagrange_multipliers)):
            prediction = None

            if method == "integrate":
                # Compute the density on a regular grid in the transformed space.
                # We do this by flowing backward and tracking the Jacobian determinant
                # (determinant = 1 for linear transformations).
                int_grid_xmax = xmax  # integration limits
                int_res = res  # integration grid resolution
                grid_shape = (len(self.bin_coords), int_res)
                grid_coords = [
                    self.bin_coords,
                    np.linspace(-int_grid_xmax, +int_grid_xmax, int_res),
                ]
                grid_points = get_grid_points(grid_coords)
                grid_points_in = self.transforms[i].inverse(grid_points)
                prob_in = self.prob(grid_points_in)

                # Assume det(J) = 1 for now; in the future, we could add a `forward_and_log_det`
                # method to `transform` object to track the Jacobian determinant.
                prob_out = prob_in

                # Compute the projection in the transformed space.
                prob_out = np.reshape(prob_out, grid_shape)
                prediction = np.sum(prob_out, axis=1)

            elif method == "particles":
                # Compute the projections using particles.
                x = self.sample(n_samples)
                x = self.transforms[i](x)
                hist, _ = np.histogram(x[:, 0], bins=self.bin_edges)
                prediction = hist
            else:
                raise ValueError("Invalid method.")

            # Normalize the one-dimensional projection.
            normalization = np.sum(prediction) * (self.bin_edges[1] - self.bin_edges[0])
            prediction = prediction / normalization

            # Update the lagrange multipliers for this projection.
            measurement = self.measurements[i]
            for j in range(len(self.lagrange_multipliers[i])):
                mij_meas = measurement[j]
                mij_pred = prediction[j]
                if (mij_meas != 0.0) and (mij_pred != 0.0):
                    self.lagrange_multipliers[i][j] *= mij_meas / mij_pred

        self.iteration += 1

    def sample(self, n: int, xmax: float = 5.0, res: int = 200):
        """Sample from discretized distribution.

        Parameters
        ----------
        n: number of samples
        xmax: grid boundary [-xmax, xmax]
        res : grid resolution

        Returns
        -------
        ndarray, shape (n, d)
        """
        coords = 2 * [np.linspace(-xmax, xmax, res)]
        prob = self.prob(get_grid_points(coords))
        prob = np.reshape(prob, (res, res))
        return sample_hist(prob, coords=coords, n=n)
