"""Methods to estimate entropy from samples."""
import typing
from typing import Optional

import numpy as np
import torch
import torch.nn as nn


class EntropyEstimator(nn.Module):
    """Estimates entropy from samples."""
    def __init__(self, prior=None) -> None:
        super().__init__()
        self.prior = prior

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class EmptyEntropyEstimator(EntropyEstimator):
    """Returns zero."""
    def __init__(self, prior=None):
        super().__init__(prior=prior)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 0.0


class CovarianceEntropyEstimator(EntropyEstimator):
    """Estimates entropy from covariance matrix."""
    def __init__(self, prior=None, pad=1.00e-12):
        if prior is not None:
            raise ValueError("This class cannot estimate relative entropy (prior != None).")
        super().__init__(prior=prior)
        self.pad = pad

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        eps = torch.sqrt(torch.det(torch.cov(x.T)))
        H = -3.0 * np.log(2.0 * np.pi * np.e) - torch.log(eps + self.pad)
        return H


class KNNEntropyEstimator(EntropyEstimator):
    """Estimates entropy from k nearest neighbors."""
    def __init__(self, prior=None, k=5):
        if prior is not None:
            raise ValueError("This class cannot estimate relative entropy (prior != None).")
        super().__init__(prior=prior)
        self.k = k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError