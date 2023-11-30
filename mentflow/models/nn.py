import torch    
import torch.nn as nn


def get_activation(activation):
    activations = {
        "relu": nn.ReLU(),
        "tanh": nn.Tanh(),
    }
    return activations[activation]
    

class NNGenerator(nn.Module):
    def __init__(
        self,
        input_features: int = 2,
        output_features: int = 2,
        hidden_layers: int = 3,
        hidden_units: int = 64,
        dropout: float =0.0,
        activation="tanh",
    ) -> None:
        activation = get_activation(activation)
        
        super().__init__()
        layer_sequence = [nn.Linear(input_features, hidden_units), activation]
        for i in range(hidden_layers):
            layer_sequence.append(nn.Linear(hidden_units, hidden_units))
            layer_sequence.append(nn.Dropout(dropout))
            layer_sequence.append(activation)
        layer_sequence.append(nn.Linear(hidden_units, output_features))
        self.stack = nn.Sequential(*layer_sequence)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.stack(x)
