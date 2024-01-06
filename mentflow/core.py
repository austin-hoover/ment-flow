import typing
from typing import Iterator
from typing import List
from typing import Optional
from typing import Tuple
from typing import Type

import torch
import torch.nn as nn
import zuko

from mentflow.entropy import EntropyEstimator
from mentflow.loss import get_loss_function
from mentflow.types_ import Model
from mentflow.types_ import PriorDistribution
from mentflow.types_ import TrainableDistribution
from mentflow.utils import unravel


class MENTFlow(Model, nn.Module):
    """Flow-based maximum entropy tomography (MENT) solver."""
    def __init__(
        self,
        generator: TrainableDistribution,
        entropy_estimator: EntropyEstimator,
        prior: Type[nn.Module],
        transforms: List[Type[nn.Module]],
        diagnostics: List[Type[nn.Module]],
        measurements: List[List[torch.Tensor]],
        discrepancy_function: str = "kld",
        penalty_parameter: float = 10.0,
    ) -> None:
        """Constructor.

        Parameters
        ----------
        generator : TrainableDistribution
            A trainable model that generates samples and evalutes the probability density.
        entropy_estimator : EntropyEstimator
            Estimates entropy from samples and probability density.
        prior : Type[nn.Module]
            A prior distribution for relative entropy estimates. If None, use absolute entropy.
        transforms : list[nn.Module], shape (n_meas,)
            The ith transform is applied before the ith measurement.
        diagnostics : list[torch.nn.Module]
            Diagnostics used to measure the distribution after each transform. 
        measurements : list[list[tensor], shape (len(diagnostics))], shape (len(transforms),)
            The measurement data.
        discrepancy_function : {"kld", "mae", "mse"}
            Function used to estimate discrepancy between simulated and measured projections.
            - "kld": KL divergence
            - "mae": mean absolute error
            - "mse": mean squared error
        penalty_parameter : float
            Loss = H + penalty_parameter * |D|.
        """
        super().__init__()
        self.generator = generator
        self.entropy_estimator = entropy_estimator
        self.set_prior(prior)
        self.set_diagnostics(diagnostics)
        self.set_measurements(measurements)
        self.transforms = transforms
        self.discrepancy_function = get_loss_function(discrepancy_function)
        self.penalty_parameter = penalty_parameter

    def set_diagnostics(self, diagnostics: List[Type[nn.Module]]):
        self.diagnostics = diagnostics
        if self.diagnostics is None:
            self.diagnostics = []
        self.n_diag = len(self.diagnostics)

    def set_measurements(self, measurements: List[List[torch.Tensor]]):
        self.measurements = measurements
        if self.measurements is None:
            self.measurements = []
        self.n_meas = len(self.measurements)

    def set_prior(self, prior):
        self.prior = prior
        if self.entropy_estimator is not None:
            self.entropy_estimator.prior = prior

    def discrepancy(self, predictions) -> List[torch.Tensor]:
        """Compute simulation-measurement discrepancy vector.

        Parameters
        ----------
        predictions : list[tensor], shape (n_meas, n_diag)
            Predictions.

        Returns
        -------
        list[tensor], shape (n_meas * n_diag,)
            The discrepancy vector.
        """
        discrepancy_vector = []
        for prediction, measurement in zip(unravel(predictions), unravel(self.measurements)):
            discrepancy = self.discrepancy_function(prediction, measurement)
            discrepancy_vector.append(discrepancy)
        return discrepancy_vector

    def sample(self, n: int) -> torch.Tensor:
        """Sample n points."""
        return self.generator.sample(n)

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the log probability at x."""
        return self.generator.log_prob(x)

    def sample_and_log_prob(self, n: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample n points and return the log probability at each point."""
        raise self.generator.sample_and_log_prob(x)

    def sample_and_entropy(self, n: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample n points and estimate the entropy from the samples."""
        x, log_prob = self.generator.sample_and_log_prob(n)
        H = self.entropy_estimator(x, log_prob)
        return (x, H)

    def simulate(self, x: torch.Tensor) -> List[List[torch.Tensor]]:
        """Simulate the measurements.

        Parameters
        ----------
        x : torch.tensor, shape (n, d)
            Particle coordinates.

        Returns
        -------
        predictions : list[list[tensor], shape (n_diag)], shape (n_meas,)
            The simulated measurements.
        """
        return simulate(x, self.transforms, self.diagnostics)

    def loss(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """Estimate the loss from a new batch.

        Parameters
        ----------
        batch_size : int
            Number of particles to samples.

        Returns
        -------
        L : tensor
            L = H + mu * |D|, where H is the entropy, D is the discrepancy vector, and |.| is
            the l1 norm.
        H : tensor
            Estimated entropy. If `self.prior` is not None, this is the negative relative entropy,
            or the KL divergence between the model and prior distribution. Otherwise we estimate
            the absolute entropy.
        D : tensor, shape (n_meas * n_diag,)
            The discrepancy vector.
        """
        x, H = self.sample_and_entropy(batch_size)
        predictions = self.simulate(x)        
        D = self.discrepancy(predictions)
        L = H + self.penalty_parameter * (sum(D) / len(D))
        return (L, H, D)

    def parameters(self) -> Iterator[nn.Parameter]:
        return self.generator.parameters()
    
    def save(self, path) -> None:
        state = {
            "generator": self.generator.state_dict(),
            "entropy_estimator": self.entropy_estimator,
            "prior": self.prior,
            "transforms": self.transforms,
            "diagnostics": self.diagnostics,
            "measurements": self.measurements,
        }
        torch.save(state, path)

    def load(self, path, device) -> None:
        state = torch.load(path, map_location=device)
        try:
            self.generator.load_state_dict(state["generator"])
        except RuntimeError:
            raise RuntimeError("Error loading generative model. Architecture mismatch?")

        self.entropy_estimator = state["entropy_estimator"]
        self.set_prior(state["prior"])
        self.transforms = state["transforms"]
        self.diagnostics = state["diagnostics"]
        self.measurements = state["measurements"]
        self.to(device)

    def to(self, device):
        if self.transforms is not None:
            for i in range(len(self.transforms)):
                self.transforms[i] = self.transforms[i].to(device)
        if len(self.diagnostics) > 0:
            for j in range(len(self.diagnostics)):
                self.diagnostics[j] = self.diagnostics[j].to(device)
        if self.generator is not None:
            self.generator = self.generator.to(device)
        return self


def simulate(x, transforms, diagnostics):        
    predictions = []
    for transform in transforms:
        x_out = transform(x)
        predictions.append([diagnostic(x_out) for diagnostic in diagnostics])
    return predictions


def simulate_nokde(x, transforms, diagnostics):
    settings = []
    for diagnostic in diagnostics:
        settings.append(diagnostic.kde)
        diagnostic.kde = False
        
    predictions = simulate(x, transforms, diagnostics)

    for setting, diagnostic in zip(settings, diagnostics):
        diagnostic.kde = setting

    return predictions
    

