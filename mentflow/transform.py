"""Differentiable transformations.

See the bmad-x repository for beam physics modeling capabilities.
(https://github.com/bmad-sim/Bmad-X)
"""
import math
import typing
import weakref
from typing import List
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn


class Transform(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def ladj(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward_and_ladj(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        y = self.forward(x)
        ladj = self.ladj(x, y)
        return (y, ladj)

    def inverse_and_ladj(self, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.inverse(y)
        ladj = -self.ladj(y, x)
        return (x, ladj)
        
        
class Linear(Transform):
    def __init__(self, matrix: torch.Tensor) -> None:
        super().__init__()
        self.set_matrix(matrix)

    def set_matrix(self, matrix: torch.Tensor) -> None:
        self.matrix = matrix
        self.matrix_inv = torch.linalg.inv(self.matrix)
        self.d = int(self.matrix.shape[0])
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.linear(x, self.matrix)

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        return nn.functional.linear(y, self.matrix_inv)

    def ladj(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.zeros(x.shape[0])

    def to(self, device):
        self.matrix = self.matrix.to(device)
        self.matrix_inv = self.matrix_inv.to(device)
        return self    


class CompositeTransform(nn.Module):
    def __init__(self, transforms) -> None:
        super().__init__()
        self.transforms = nn.Sequential(transforms)

    def forward_and_ladj(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        ladj = 0.0
        for transform in self.transforms:
            x, _ladj = transform.forward_and_ladj(x)
            ladj += _ladj
        return (x, ladj)

    def inverse_and_ladj(self, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        ladj = 0.0
        for transform in self.transforms:
            y, _ladj = transform.inverse_and_ladj(y)
            ladj += _ladj
        return (y, ladj)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for transform in self.transforms:
            x = transform(x)
        return x        

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        for transform in self.transforms:
            y = transform.inverse(y)
        return y

    def to(self, device):
        for transform in self.transforms:
            transform.to(device)
        return self


def Projection1D(Transform):
    def __init__(self, v: torch.Tensor):
        super().__init__()
        self.v = v

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sum(x * self.v, dim=1)


def rotation_matrix(angle):
    _cos = np.cos(angle)
    _sin = np.sin(angle)
    return torch.tensor([[_cos, _sin], [-_sin, _cos]])
