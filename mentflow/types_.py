"""Source: https://github.com/lollcat/fab-torch/blob/master/fab/types_.py"""
import abc
import typing
from typing import Any
from typing import Callable
from typing import Iterator
from typing import Mapping
from typing import Tuple

import torch


class Distribution(abc.ABC):
    @abc.abstractmethod
    def sample(self, size: int) -> torch.Tensor:
        raise NotImplementedError

    @abc.abstractmethod
    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @abc.abstractmethod
    def sample_and_log_prob(self, size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError


class PriorDistribution(typing.Protocol):
    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        ...


class Model:
    @abc.abstractmethod
    def loss(self, batch_size: int) -> torch.Tensor:
        raise NotImplementedError

    def get_iter_info(self) -> Mapping[str, Any]:
        raise NotImplementedError

    def get_eval_info(self, inner_batch_size: int) -> Mapping[str, Any]:
        raise NotImplementedError

    @abc.abstractmethod
    def parameters(self) -> Iterator[torch.nn.Parameter]:
        raise NotImplementedError
        
    def save(self, path: str) -> None:
        raise NotImplementedError

    def load(self, path: str, map_location=None) -> None:
        raise NotImplementedError
        