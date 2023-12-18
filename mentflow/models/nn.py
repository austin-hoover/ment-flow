"""Non-invertible neural network generator.

References
----------
[1] https://doi.org/10.1103/PhysRevLett.130.145001
"""
from typing import Tuple

import torch    
import torch.nn as nn

from mentflow.types_ import TrainableDistribution


def get_activation(activation):
    activations = {
        "relu": nn.ReLU(),
        "tanh": nn.Tanh(),
    }
    return activations[activation]
    

class NNTransformer(nn.Module):
    def __init__(
        self,
        input_features: int = 2,
        output_features: int = 2,
        hidden_layers: int = 3,
        hidden_units: int = 64,
        dropout: float = 0.0,
        activation="tanh",
    ) -> None:
        activation = get_activation(activation)
        
        super().__init__()
        layers = [nn.Linear(input_features, hidden_units), activation]
        for i in range(hidden_layers):
            layers.append(nn.Linear(hidden_units, hidden_units))
            layers.append(nn.Dropout(dropout))
            layers.append(activation)
        layers.append(nn.Linear(hidden_units, output_features))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class NNGenerator(TrainableDistribution):
    def __init__(self, base=None, transformer=None) -> None:
        super().__init__()
        self.transformer = transformer
        self.base = base

    def sample(self, n: int) -> torch.Tensor:
        x = self.base.rsample((n,))
        x = self.transformer(x)
        return x

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        return None

    def sample_and_log_prob(self, n: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.sample(n)
        log_prob = None
        return (x, log_prob)

    def to(self, device):
        self.transformer = self.transformer.to(device)
        return self

        
        