"""Iterative maximum-entropy tomography (MENT) solvers.

References
----------
[1] https://doi.org/10.1103/PhysRevAccelBeams.25.042801
"""
import numpy as np
import scipy.interpolate

from mentflow.utils import get_grid_points
from mentflow.utils import sample_hist


class MENT_2D:
    """2D MENT reconstruction model.

    TODO
    ----
    - Fix the update step. The current version is not completely converging.
    - Use torch.
    - Match to upstream stucture for easier comparison.
    """
    def __init__(self, transfer_matrices=None, measurements=None, edges=None, prior=None):
        self.transfer_matrices = transfer_matrices
        self.measurements = measurements
        self.edges = edges
        self.coords = [0.5 * (e[:-1] + e[1:]) for e in edges]
        
        self.prior = prior
        if self.prior is None:
            self.prior = UniformDistribution(scale=100.0)
            
        self.lagrange_multipliers = []
        for i, measurement in enumerate(self.measurements):
            self.lagrange_multipliers.append([])
            for j in range(len(measurement)):
                mij = float(measurement[j] > 0)
                self.lagrange_multipliers[-1].append(mij)
            
    def chi(self, x, i=0, j=0):
        """Return chi function at points x (projection i, bin j)."""
        idx = np.logical_and(
            x[:, 0] >= self.edges[i][j],
            x[:, 0] < self.edges[i][j + 1],
        )
        return idx.astype(float)

    def prob(self, x, exclude=-1):
        """Return probability density at points x."""
        idx = list(range(len(self.measurements)))
        if exclude >= 0:
            idx = [i for i in idx if i != exclude]
    
        _prob = np.zeros(len(x))
        for i in idx:
            x_out = np.matmul(x, self.transfer_matrices[i].T)
            for j in range(len(self.measurements[i])):
                _prob += self.chi(x_out, i, j) * self.lagrange_multipliers[i][j]
        return _prob * self.prior.prob(x)
    
    def step(self):
        """Update the lagrange multipliers."""
        # Define a grid at the reconstruction location.
        xmax = 4.0
        res = 150
        grid_coords = 2 * [np.linspace(-xmax, xmax, res)] 
        grid_points = get_grid_points(grid_coords)

        for i in range(len(self.lagrange_multipliers)):
            # Compute the integrand using the existing lagrange multipliers.
            prob = self.prob(grid_points, exclude=i)
            matrix = self.transfer_matrices[i]
            grid_points_out = np.matmul(grid_points, matrix.T)
            prob_out = prob  # unit determinant

            # Interpolate the density on a regular grid at the measurement location.
            int_res = 250  # interpolation grid resolution
            grid_coords_int = [self.coords[i], np.linspace(-xmax, xmax, int_res)]
            grid_points_int = get_grid_points(grid_coords_int)
            prob_out_int = scipy.interpolate.griddata(
                grid_points_out, prob_out, grid_points_int, method='linear', fill_value=0.0
            )
            prob_out_int = np.reshape(prob_out_int, [len(c) for c in grid_coords_int])
            normalization = np.sum(prob_out_int) * (self.edges[i][1] - self.edges[i][0])
            prob_out_int = prob_out_int / normalization

            # Compute the projection.
            prediction = np.sum(prob_out_int, axis=1)
            normalization = np.sum(prediction) * (self.edges[i][1] - self.edges[i][0])
            prediction = prediction / normalization

            # Update the lagrange multipliers for this projection.
            measurement = self.measurements[i]
            for j in range(len(self.lagrange_multipliers[i])):
                mij_meas = measurement[j]
                mij_pred = prediction[j]
                factor = 1.0
                if mij_meas != 0:
                    factor = mij_meas / mij_pred
                self.lagrange_multipliers[i][j] = self.lagrange_multipliers[i][j] * factor

    def sample(self, n, xmax=5.0, res=200):
        """Sample from the distribution."""
        coords = 2 * [np.linspace(-xmax, xmax, res)]
        prob = self.prob(get_grid_points(coords))
        prob = np.reshape(prob, (res, res))
        return sample_hist(prob, coords=coords, n=n)
                

class GaussianDistribution:
    """Gaussian prior distribution."""
    def __init__(self, scale=1.0):
        self.scale = scale
    
    def prob(self, x):
        return np.exp(-(1.0 / (2.0 * self.scale**2)) * np.sum(np.square(x), axis=1)) / (self.scale * np.sqrt(2.0 * np.pi))


class UniformDistribution:
    """Uniform prior distribution."""
    def __init__(self, scale=100.0, d=2):
        self.scale = scale
        self.d = d
        self.volume = (100.0)**self.d
    
    def prob(self, x):
        return np.full(len(x), 1.0 / self.volume)
