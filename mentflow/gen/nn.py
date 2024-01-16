"""Neural network generator.

References
----------
[1] https://doi.org/10.1103/PhysRevLett.130.145001
"""
from typing import Tuple, List

import torch    
import torch.nn as nn

from .gen import GenModel


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
        hidden_layers: int = 2,
        hidden_units: int = 20,
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


class NNGen(GenModel):
    def __init__(self, base=None, network=None) -> None:
        super().__init__()
        self.network = network
        self.base = base

    def sample(self, n: int) -> torch.Tensor:
        x = self.base.rsample((n,))
        x = self.network(x)
        return x

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        return None

    def sample_and_log_prob(self, n: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.sample(n)
        log_prob = None
        return (x, log_prob)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.network(z)

    def forward_steps(self, z: torch.Tensor) -> List[torch.Tensor]:
        return [z, self.network(z)]

    def sample_base(self, n: int) -> torch.Tensor:
        return self.base.rsample((n,))

    def to(self, device):
        self.network = self.network.to(device)
        # Need to send base to device... 
        return self

        
        