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
    """Base class for probability distributions."""    
    @abc.abstractmethod
    def sample(self, n: int) -> torch.Tensor:
        """Sample n points."""
        raise NotImplementedError

    @abc.abstractmethod
    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the log probability at x."""
        raise NotImplementedError

    @abc.abstractmethod
    def sample_and_log_prob(self, n: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample n points and compute the log-probability at each point."""
        raise NotImplementedError


class PriorDistribution(typing.Protocol):
    """Protocol class for distributions only requiring log_prob."""
    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        ...


class Model:
    """Base class for trainable models."""
    @abc.abstractmethod
    def loss(self, batch_size: int) -> torch.Tensor:
        """Compute the loss."""
        raise NotImplementedError

    def get_iter_info(self) -> Mapping[str, Any]:
        """Return information from the latest iteration."""
        raise NotImplementedError

    def get_eval_info(self, inner_batch_size: int) -> Mapping[str, Any]:
        """Evaluate the model at the current point in training. This is useful for more
        expensive evaluation metrics than what is computed in get_iter_info."""
        raise NotImplementedError

    @abc.abstractmethod
    def parameters(self) -> Iterator[torch.nn.Parameter]:
        """Return the model parameters."""

    def save(self, path) -> None:
        """Save model to file_path."""
        raise NotImplementedError

    def load(self, path, map_location) -> None:
        """Load model from file_path."""
        raise NotImplementedError
        