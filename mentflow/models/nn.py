import torch    
import torch.nn as nn


class NNTransformer(nn.Module):
    """Neural network transformer.
    
    Source: Roussel et al., "Phase Space Reconstruction from Accelerator Beam 
    Measurements Using Neural Networks and Differentiable Simulations", PRL (2022).
    """
    def __init__(
        self,
        features: int = 2,
        hidden_layers: int = 3,
        hidden_units: int = 64,
        dropout: float =0.0,
        activation="tanh",
    ) -> None:
        activations = {
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
        }
        activation = activations[activation]
        
        super().__init__()
        layer_sequence = [nn.Linear(features, hidden_units), activation]
        for i in range(hidden_layers):
            layer_sequence.append(nn.Linear(hidden_units, hidden_units))
            layer_sequence.append(nn.Dropout(dropout))
            layer_sequence.append(activation)
        layer_sequence.append(nn.Linear(hidden_units, features))
        self.stack = nn.Sequential(*layer_sequence)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.stack(x)
