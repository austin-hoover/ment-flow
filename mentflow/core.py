"""Maximum entropy tomography using normalizing flows."""
from typing import Any
from typing import Iterator
from typing import List
from typing import Optional
from typing import Tuple
from typing import Type

import torch
import torch.nn as nn
import zuko

import mentflow.losses as losses


class GenMENT(nn.Module):
    """Generative maximum-entropy tomography (MENT) solver."""
    def __init__(
        self,
        d : int,
        flow: Type[nn.Module],
        target: Type[torch.distributions.Distribution],
        lattices: List[Type[nn.Module]],
        diagnostic: Type[nn.Module],
        measurements: List[torch.Tensor],
        discrepancy_function: str = "kld",
        penalty_parameter: float = 10.0,
    ) -> None:
        """Constructor.

        Parameters
        ----------
        d : int
            Dimensionality of the space.
        flow : torch.nn.Module subclass
            A differentiable model that generates samples and (possibly) evalutes the 
            probability density.
        target : Distribution
            A prior distribution for relative entropy calculations. It must implement
            `log_prob(x)`. If None, the absolute entropy is used.
        lattices : list[Lattice], shape (n_meas,)
            A list of lattices representing the transformation before each measurement.
        diagnostic : torch.nn.Module
            The diagnostic used to measure the distribution after passing through the 
            lattice. Currently, only one diagnostic can be added.
        measurements : list[tensor], shape (n_meas,)
            The measurement data.
        penalty_parameter : float
            Penalty parameter for the loss function (loss = entropy + penalty_parameter * discrepancy).
        discrepancy_function : {"kld", "mae", "mse"}
            Function used to estimate discrepancy between simulated and measured projections.
            - "kld": KL divergence
            - "mae": mean absolute error
            - "mse": mean squared error
            - "wasserstein": Wasserstein distance
        """
        super().__init__()
        self.d = d
        self.flow = flow
        self.target = target
        self.lattices = lattices
        self.diagnostic = diagnostic
        self.set_measurements(measurements)
        self.penalty_parameter = penalty_parameter
        self.discrepancy_function = {
            "mae": losses.mae,
            "mse": losses.mse,
            "kld": losses.kld,
        }[discrepancy_function]

    def set_measurements(self, measurements: List[torch.Tensor]):
        self.n_meas = 0
        if measurements:
            self.n_meas = len(measurements)
        self.measurements = measurements

    def simulate(self, x: torch.Tensor, **kws) -> List[torch.Tensor]:
        """Simulate the measurements.

        Parameters
        ----------
        x : tensor, shape (n, d)
            Particle coordinates.
        **kws
            Key word arguments for `self.diagnostic.forward` method.

        Returns
        -------
        predictions : list[tensor], shape (n_meas,)
            The simulated measurements.
        """
        predictions = []
        for lattice in self.lattices:
            prediction = self.diagnostic(lattice(x), **kws)
            predictions.append(prediction)
        return predictions

    def discrepancy(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Simulate the measurements and compute discrepancy with actual measurements.

        Parameters
        ----------
        x : tensor, shape (n, d)
            Particle coordinates.

        Returns
        -------
        C : list, shape (n_meas,)
            The discrepancy vector.
        """
        predictions = self.simulate(x)
        C = self.n_meas * [0.0]
        for i in range(self.n_meas):
            C[i] = self.discrepancy_function(predictions[i], self.measurements[i])
        return C

    def loss(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """Compute the Penalty Method (PM) loss using a new batch.
        
        L = H + mu * |C| / n_meas, where H is the entropy, C is the discrepancy vector, 
        and |.| is the l1 norm.

        Parameters
        ----------
        batch_size : int
            Number of particles to samples.

        Returns
        -------
        L : tensor
        H : tensor
            Estimated entropy. If `self.target` is not None, this is the negative relative entropy,
            or the KL divergence between the model and target distribution. Otherwise we estimate
            the absolute entropy.
        C : tensor, shape (n_meas,)
            The discrepancy vector.
        """
        x, H = self.sample_and_entropy(batch_size)
        C = self.discrepancy(x)
        L = H + self.penalty_parameter * (sum(C) / self.n_meas)
        return (L, H, C)

    def parameters(self) -> Iterator[nn.Parameter]:
        """Return trainable parameters."""
        return self.flow.parameters()

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the log probability at x."""
        raise NotImplementedError

    def sample(self, n: int) -> torch.Tensor:
        """Sample n points.."""
        raise NotImplementedError

    def sample_and_log_prob(self, n: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample n points and the log probability at each point."""
        raise NotImplementedError

    def sample_and_entropy(self, n: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample n points and estimate the entropy from the samples."""
        raise NotImplementedError

    def save(self, path: str) -> None:
        """Save the model parameters to a binary file."""
        state = {
            "d": self.d,
            "flow": self.flow.state_dict(),
            "target": self.target,
            "lattices": self.lattices,
            "diagnostic": self.diagnostic,
            "measurements": self.measurements,
        }
        torch.save(state, path)

    def load(self, path: str, device: Optional[torch.device] = None) -> None:
        """Load the model parameters from a binary file."""
        state = torch.load(path, map_location=device)        
        try:
            self.flow.load_state_dict(state["flow"])
        except RuntimeError:
            raise RuntimeError("Error loading flow. Architecture mismatch?")
        self.target = state["target"]
        self.lattices = state["lattices"]
        self.diagnostic = state["diagnostic"]
        self.measurements = state["measurements"]
        self.to(device)

    def to(self, device):
        """Set the torch device."""
        if self.lattices is not None:
            for i in range(len(self.lattices)):
                self.lattices[i] = self.lattices[i].to(device)
        if self.diagnostic is not None:
            self.diagnostic = self.diagnostic.to(device)
        if self.flow is not None:
            self.flow = self.flow.to(device)
        return self


class MENTFlow(GenMENT):
    """GenMENT model using normalizing flow (invertible neural network)."""
    def __init__(self, flow: zuko.flows.Flow, **kws) -> None:
        """Constructor.

        Parameters
        ----------
        flow : zuko.flows.Flow
            A normalizing flow model. We use the `zuko` package (https://github.com/probabilists/zuko).
            The base distribution is defined within the flow object.
        **kws
            Key word arguments passed `GenMENT`.
        """
        super().__init__(flow=flow, **kws)

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        return self.flow().log_prob(x)

    def sample(self, n: int) -> torch.Tensor:
        x = self.flow().rsample((n,))
        return x

    def sample_and_log_prob(self, n: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x, log_prob = self.flow().rsample_and_log_prob((n,))
        return (x, log_prob)

    def sample_and_entropy(self, n: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x, log_prob = self.sample_and_log_prob(n)
        H = torch.mean(log_prob)
        if self.target is not None:
            H = H - torch.mean(self.target.log_prob(x))
        return (x, H)


class MENTNN(GenMENT):
    """GenMENT model using non-invertible neural network."""
    def __init__(
        self,
        flow: Type[nn.Module],
        base: Optional[Type[torch.distributions.Distribution]] = None,
        entropy_estimator: Optional[Type[nn.Module]] = None,
        **kws
    ) -> None:
        """Constructor.

        Parameters
        ----------
        flow : torch.nn.Module
            A feedforward neural network. The network transforms a d'-dimensional base 
            distribution to a d-dimensional data distribution.
        base : torch.distributions.Distribution
            The base distribution. Defaults to d-dimensional Gaussian distribution.
        entropy_estimator : torch.nn.Module
            Estimates the (relative) entropy from samples.
        **kws
            Key word arguments passed `GenMENT`.
        """
        super().__init__(flow=flow, **kws)
        self.entropy_estimator = entropy_estimator
        self.base = base
        if self.base is None:
            d = self.d
            self.base = torch.distributions.Normal(loc=torch.zeros(d), scale=torch.ones(d))

    def sample(self, n: int) -> torch.Tensor:
        x = self.base.rsample((n,))
        x = self.flow(x)
        return x

    def sample_and_entropy(self, n: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.sample(n)
        H = self.entropy_estimator(x, target=self.target)
        return (x, H)
