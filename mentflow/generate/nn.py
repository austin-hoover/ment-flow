"""Neural network generator.

References
----------
[1] https://doi.org/10.1103/PhysRevLett.130.145001
"""
from typing import Any
from typing import Callable
from typing import List
from typing import Tuple

import torch    
import torch.nn as nn

from .base import GenerativeModel


def get_activation(name: bool) -> Callable:
    if name == "relu":
        return nn.ReLU()
    elif name == "tanh":
        return nn.Tanh()
    else:
        raise ValueError(f"Invalid activation '{name}'")
        

class NNTransform(nn.Module):
    def __init__(
        self,
        input_features: int = 2,
        output_features: int = 2,
        hidden_layers: int = 2,
        hidden_units: int = 20,
        dropout: float = 0.0,
        activation: float = "tanh",
    ) -> None:
        activation = get_activation(activation)
        
        super().__init__()
        layers = [nn.Linear(input_features, hidden_units), activation]
        for _ in range(hidden_layers):
            layers.append(nn.Linear(hidden_units, hidden_units))
            layers.append(nn.Dropout(dropout))
            layers.append(activation)
        layers.append(nn.Linear(hidden_units, output_features))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class NNGenerator(GenerativeModel):
    def __init__(self, base: Any, transform: nn.Module) -> None:
        super().__init__()
        self.base = base
        self.transform = transform

    def sample(self, n: int) -> torch.Tensor:
        x = self.base.rsample((n,))
        x = self.transform(x)
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

        
        