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
    """Estimates entropy from k nearest neighbors."""
    def __init__(self, k: int = 5) -> None:
        super().__init__()
        self.k = k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 0.0