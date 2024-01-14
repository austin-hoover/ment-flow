"""Tools for tomographic simulations."""
from typing import List
from typing import Type
import torch
import torch.nn as nn


def forward(
    x: torch.Tensor, 
    transforms: List[Type[nn.Module]],
    diagnostics: List[List[Type[nn.Module]]]
) -> List[List[torch.Tensor]]:
    """Forward process.

    Parameters
    ----------
    x : tensor, shape (n, d)
        Input particle coordinates.
    transforms : list[Transform],
        A list of transforms.
    diagnostics : list[list[Diagnostic]],
        diagnostics[i] is a list of diagnostics to apply after transform i.

    Returns
    -------
    list[list[tensor]]
        Predictions for each transform and diagnostic. 
    """
    predictions = []
    for index, transform in enumerate(transforms):
        u = transform(x.clone())
        predictions.append([diagnostic(u) for diagnostic in diagnostics[index]])
    return predictions


class Simulator:
    """Manages forward process simulations."""
    def __init__(
        self,         
        transforms: List[Type[nn.Module]],
        diagnostics: List[List[Type[nn.Module]]],
    ) -> None:
        self.transforms = transforms
        self.diagnostics = diagnostics
            
    def forward(self, x: torch.Tensor) -> List[List[torch.Tensor]]:
        return forward(x, self.transforms, self.diagnostics)

