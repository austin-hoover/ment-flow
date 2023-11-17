import numpy as np
import scipy.interpolate

from mentflow.utils import get_grid_points
from mentflow.utils import sample_hist


class MENT:
    """2D MENT reconstruction model.

    This does not seed to fit the data at the moment, although it gets close
    depending on the distribution. Work in progres..
    """
    def __init__(self, transfer_matrices=None, measurements=None, edges=None, prior=None):
        self.transfer_matrices = transfer_matrices
        self.measurements = measurements
        self.edges = edges
        self.coords = [0.5 * (e[:-1] + e[1:]) for e in edges]
        
        self.prior = prior
        if self.prior is None:
            self.prior = Uniform(scale=100.0)
            
        self.lagrange_multipliers = []
        for i, measurement in enumerate(self.measurements):
            self.lagrange_multipliers.append([])
            for j in range(len(measurement)):
                mij = float(measurement[j] > 0)
                self.lagrange_multipliers[-1].append(mij)

    def transport(self, z, i=0):
        return np.apply_along_axis(lambda row: np.matmul(self.transfer_matrices[i], row), 1, z)
            
    def _chi(self, z, i=0, j=0):
        return float(self.edges[i][j] <= z[0] < self.edges[i][j + 1])

    def _chi_vectorized(self, z, i=0, j=0):
        idx = np.logical_and(
            z[:, 0] >= self.edges[i][j],
            z[:, 0] < self.edges[i][j + 1],
        )
        return idx.astype(float)

    def prob(self, z):
        _prob = 0.0
        for i, (measurement, matrix) in enumerate(zip(self.measurements, self.transfer_matrices)):
            z_out = np.matmul(matrix, z)
            for j in range(len(measurement)):
                _prob += self._chi(z_out, i, j) * self.lagrange_multipliers[i][j]
        return _prob * self.prior.prob(z)

    def prob_vectorized(self, z):
        _prob = np.zeros(len(z))
        for i, (measurement, matrix) in enumerate(zip(self.measurements, self.transfer_matrices)):
            z_out = self.transport(z, i=i)
            for j in range(len(measurement)):
                _prob += self._chi_vectorized(z_out, i, j) * self.lagrange_multipliers[i][j]
        return _prob * self.prior.prob_vectorized(z)
        
    def forward(self, res=100, xmax=4.0):
        """Simulate the measurements by integrating the distribution function."""        
        grid_coords = 2 * [np.linspace(-xmax, xmax, res)]
        grid_points = get_grid_points(grid_coords)
        
        prob = np.zeros(grid_points.shape[0])
        for k in range(grid_points.shape[0]):
            prob[k] = self.prob(grid_points[k, :])

        predictions = []
        for i, matrix in enumerate(self.transfer_matrices):
            grid_points_out = self.transport(grid_points, i=i)
            prob_out = prob  # unit determinant
            
            grid_coords_int = [self.coords[i], np.linspace(-xmax, xmax, 500)]
            grid_points_int = get_grid_points(grid_coords_int)
            
            prob_out_int = scipy.interpolate.griddata(grid_points_out, prob_out, grid_points_int, method='linear', fill_value=0.0)
            prob_out_int = np.reshape(prob_out_int, [len(c) for c in grid_coords_int])
            prob_out_int = prob_out_int / np.sum(prob_out_int) / (self.edges[i][1] - self.edges[i][0])
            
            prediction = np.sum(prob_out_int, axis=1)
            prediction = prediction / np.sum(prediction) / (self.edges[i][1] - self.edges[i][0])
            predictions.append(prediction)
        return predictions 

    def forward_samples(self, n=10000, xmax=8.0, res=200):
        z = self.sample(n, xmax=xmax, res=res)
        predictions = []
        for i, matrix in enumerate(self.transfer_matrices):
            z_out = self.transport(z, i=i)
            hist, _ = np.histogram(z_out[:, 0], bins=self.edges[i])
            prediction = hist / np.sum(hist) / (self.edges[i][1] - self.edges[i][0])
            predictions.append(prediction)
        return predictions

    def sample(self, n, xmax=5.0, res=200):
        coords = 2 * [np.linspace(-xmax, xmax, res)]
        prob = self.prob_vectorized(get_grid_points(coords))
        prob = np.reshape(prob, (res, res))
        return sample_hist(prob, coords=coords, n=n)

    def step(self, predictions):
        for i in range(len(self.lagrange_multipliers)):
            for j in range(len(self.lagrange_multipliers[i])):
                mij_meas = self.measurements[i][j]
                mij_pred = predictions[i][j]
                factor = 1.0
                if mij_meas != 0:
                    factor = mij_meas / mij_pred
                self.lagrange_multipliers[i][j] = self.lagrange_multipliers[i][j] * factor
                

class GaussianPrior:
    def __init__(self, scale=1.0):
        self.scale = scale
    
    def prob(self, z):
        return np.exp(-(1.0 / (2.0 * self.scale**2)) * np.sum(np.square(z))) / (self.scale * np.sqrt(2.0 * np.pi))

    def prob_vectorized(self, z):
        return np.exp(-(1.0 / (2.0 * self.scale**2)) * np.sum(np.square(z), axis=1)) / (self.scale * np.sqrt(2.0 * np.pi))


class UniformPrior:
    def __init__(self, scale=100.0, d=2):
        self.scale = scale
        self.d = d
        self.volume = (100.0)**self.d
    
    def prob(self, z):
        return 1.0 / self.volume   

    def prob_vectorized(self, z):
        return np.full(len(z), 1.0 / self.volume)
