"""Methods to estimate (negative) entropy from samples and/or probability density."""
from typing import Any
from typing import Optional

import numpy as np
import torch


class EntropyEstimator(torch.nn.Module):
    """Estimates negative entropy from samples and/or log probability."""
    def __init__(self, prior: Any = None) -> None:
        super().__init__()
        self.prior = prior

    def forward(self, x: torch.Tensor, log_prob: torch.Tensor = None) -> torch.Tensor:
        raise NotImplementedError


class EmptyEntropyEstimator(EntropyEstimator):
    """Returns zero."""
    def __init__(self, prior: Any = None) -> None:
        super().__init__(prior=prior)

    def forward(self, x: torch.Tensor, log_prob: torch.Tensor = None) -> torch.Tensor:
        return 0.0


class CovarianceEntropyEstimator(EntropyEstimator):
    """Estimates negative entropy from covariance matrix."""
    def __init__(self, prior: Any = None, pad: float = 1.00e-12) -> None:
        if prior is not None:
            raise ValueError("This class cannot estimate relative entropy (prior != None).")
        super().__init__(prior=prior)
        self.pad = pad

    def forward(self, x: torch.Tensor, log_prob: torch.Tensor = None) -> torch.Tensor:
        eps = torch.sqrt(torch.det(torch.cov(x.T)))
        H = -3.0 * np.log(2.0 * np.pi * np.e) - torch.log(eps + self.pad)
        return H


class KNNEntropyEstimator(EntropyEstimator):
    """Estimates negative entropy from k nearest neighbors."""
    def __init__(self, prior: Any = None, k: int = 5) -> None:
        if prior is not None:
            raise ValueError("This class cannot estimate relative entropy (prior != None).")
        super().__init__(prior=prior)
        self.k = k

    def forward(self, x: torch.Tensor, log_prob: torch.Tensor = None) -> torch.Tensor:
        raise NotImplementedError


class MonteCarloEntropyEstimator(EntropyEstimator):
    """Estimates negative entropy from Monte Carlo."""
    def __init__(self, prior: Any = None) -> None:
        super().__init__(prior=prior)

    def forward(self, x: torch.Tensor, log_prob: torch.tensor) -> torch.Tensor:
        H = torch.mean(log_prob)
        if self.prior is not None:
            H = H - torch.mean(self.prior.log_prob(x))
        return H
        
