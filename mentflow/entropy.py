"""Methods to estimate entropy from samples."""
import numpy as np
import torch
import torch.nn as nn


class EntropyEstimator(nn.Module):
    """Estimates entropy from samples."""
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class EmptyEntropyEstimator(EntropyEstimator):
    """Returns zero."""
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 0.0


class CovarianceEntropyEstimator(EntropyEstimator):
    """Estimates entropy from covariance matrix."""
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        eps = torch.sqrt(torch.det(torch.cov(x.T)))
        H = -3.0 * np.log(2.0 * np.pi * np.e) - torch.log(eps)
        return H


class KNNEntropyEstimator(EntropyEstimator):
    """Estimates entropy from k nearest neighbors (KNN).

    TO DO: check math and fix errors.
    """
    def __init__(self, k: int = 5) -> None:
        super().__init__()
        self.k = k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d = x.shape[1]
        epsilons = torch.zeros(x.shape)
        for i in range(x.shape[0]):
            distances = x[i, :] - x[:, :]
            distances, _ = torch.sort(distances, dim=0)
            epsilons[i, :] = 2.0 * distances[self.k, :]
            hi = torch.clamp(x[i, :] + 0.5 * epsilons[i, :], None, 1.0) 
            lo = torch.clamp(x[i, :] - 0.5 * epsilons[i, :], 0.0, None)
            epsilons[i, :] = hi - lo
        H = (
            - torch.special.digamma(torch.tensor(self.k)) 
            + torch.special.digamma(torch.tensor(d)) 
            + ((d - 1) / self.k) + torch.mean(torch.sum(epsilons, 1))
        )
        return H