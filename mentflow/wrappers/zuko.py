import typing
from typing import Tuple

import torch
import zuko

from mentflow.types_ import TrainableDistribution


class WrappedZukoFlow(TrainableDistribution):
    def __init__(self, flow: zuko.flows.Flow):
        super().__init__()
        self._flow = flow

    def sample(self, n: int) -> torch.Tensor:
        return self._flow().rsample((n,))

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        return self._flow().log_prob(x)

    def sample_and_log_prob(self, n: int) -> Tuple[torch.Tensor, torch.Tensor]:
        (x, log_prob) = self._flow().rsample_and_log_prob((n,))
        return (x, log_prob)
