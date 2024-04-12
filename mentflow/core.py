from typing import Callable
from typing import Iterator
from typing import List
from typing import Tuple
from typing import Type

import torch
import torch.nn as nn
import zuko

from mentflow.entropy import EntropyEstimator
from mentflow.generate import GenerativeModel
from mentflow.simulate import forward
from mentflow.types_ import Model
from mentflow.utils import unravel


class MENTFlow(Model, nn.Module):
    """Generative maximum-entropy tomography solver."""
    def __init__(
        self,
        transforms: List[Type[nn.Module]],
        diagnostics: List[List[Type[nn.Module]]],
        measurements: List[List[torch.Tensor]],
        generator: GenerativeModel,
        prior: Type[nn.Module],
        entropy_estimator: Callable,
        discrepancy_function: str = "kld",
        penalty_parameter: float = 10.0,
    ) -> None:
        """Constructor.

        Parameters
        ----------
        transforms : list[nn.Module]
            Mappings between reconstruction and measurement locations.
        diagnostics : list[list[torch.nn.Module]
            diagnostics[i] is a list of diagnostics applied after the ith transform.
        measurements : list[list[tensor]
            Measured data corresponding to each diagnostic.
        generator : GenerativeModel
            A trainable model that generates samples and (possibly) evalutes the probability 
            density.
        entropy_estimator : Any
            Returns negative entropy from samples and/or log probability.
            Call signature: `entropy_estimator(x: torch.Tensor, log_prob: torch.Tensor) -> torch.Tensor`.
        discrepancy_function: Callable
            Computes scalar difference between measurements and predictions.
            Call signature: `discrepancy_function(pred: torch.Tensor, meas: torch.Tensor) -> torch.Tensor (float)`.
        penalty_parameter : float
            Loss = negative_entropy + penalty_parameter * mean_absolute_discrepancy.
        """
        super().__init__()
        self.transforms = transforms
        self.diagnostics = self.set_diagnostics(diagnostics)
        self.measurements = self.set_measurements(measurements)

        self.generator = generator
        self.entropy_estimator = entropy_estimator
        self.discrepancy_function = discrepancy_function
        self.penalty_parameter = penalty_parameter

    def set_diagnostics(self, diagnostics: List[Type[nn.Module]]):
        self.diagnostics = diagnostics
        if self.diagnostics is None:
            self.diagnostics = [[]]
        return self.diagnostics

    def set_measurements(self, measurements: List[List[torch.Tensor]]):
        self.measurements = measurements
        if self.measurements is None:
            self.measurements = [[]]
        return self.measurements

    def sample(self, size: int) -> torch.Tensor:
        return self.generator.sample(int(size))

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        return self.generator.log_prob(x)

    def sample_and_log_prob(self, size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.generator.sample_and_log_prob(size)

    def sample_and_entropy(self, n: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x, log_prob = self.sample_and_log_prob(n)
        H = self.entropy_estimator(x, log_prob)
        return (x, H)

    def discrepancy_vector(self, predictions: List[List[torch.Tensor]]) -> List[torch.Tensor]:
        discrepancy_vector = []
        for pred, meas in zip(unravel(predictions), unravel(self.measurements)):
            discrepancy_vector.append(self.discrepancy_function(pred, meas))
        return discrepancy_vector

    def loss(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """Estimate the loss from a new batch.

        Parameters
        ----------
        batch_size : int
            Number of particles to sample.

        Returns
        -------
        L : tensor
            L = H + mu * |D|, where H is the entropy, D is the discrepancy vector, and |.| is
            the l1 norm.
        H : tensor
            Estimated negative entropy.
        D : tensor, shape (len(measurements) * len(diagnostics))
            The discrepancy vector.
        """
        x, H = self.sample_and_entropy(batch_size)
        predictions = forward(x, self.transforms, self.diagnostics)
        D = self.discrepancy_vector(predictions)
        L = H + self.penalty_parameter * (sum(D) / len(D))
        return (L, H, D)

    def parameters(self) -> Iterator[nn.Parameter]:
        return self.generator.parameters()
    
    def save(self, path) -> None:
        state = {
            "generator": self.generator.state_dict(),
            "entropy_estimator": self.entropy_estimator,
            "transforms": self.transforms,
            "diagnostics": self.diagnostics,
            "measurements": self.measurements,
        }
        torch.save(state, path)

    def load(self, path, device=None):
        state = torch.load(path, map_location=device)
        try:
            self.generator.load_state_dict(state["gen"])
        except RuntimeError:
            raise RuntimeError("Error loading generative model. Architecture mismatch?")

        self.entropy_estimator = state["entropy_estimator"]
        self.transforms = state["transforms"]
        self.diagnostics = state["diagnostics"]
        self.measurements = state["measurements"]
        self.to(device)

    def to(self, device):
        if self.transforms is not None:
            for transform in self.transforms:
                transform = transform.to(device)
        if self.diagnostics is not None:
            for index in range(len(self.diagnostics)):
                for diagnostic in self.diagnostics[index]:
                    diagnostic = diagnostic.to(device)
        if self.measurements is not None:
            for index in range(len(self.measurements)):
                for measurement in self.measurements[index]:
                    measurement = measurement.to(device)
        if self.generator is not None:
            self.generator = self.generator.to(device)
        return self


