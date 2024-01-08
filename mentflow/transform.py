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


class CompositeTransform(nn.Module):
    def __init__(self, *transforms) -> None:
        super().__init__()
        self.transforms = nn.Sequential(*transforms)

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

    def reverse_transforms(self) -> None:
        self.transforms = nn.Sequential(*reversed([layer for layer in self.transforms]))

    def to(self, device):
        for transform in self.transforms:
            transform.to(device)
        return self


class SymplecticTransform(Transform):
    def __init__(self) -> None:
        super().__init__()

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        x = y.clone()
        x = reverse_momentum(x)
        x = self.forward(x)
        x = reverse_momentum(x)
        return x        
        
    def ladj(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.zeros(x.shape[0])
        

def CompositeSymplecticTransform(CompositeTransform):
    def __init___(self, *transforms) -> None:
        super().__init__(*transforms)

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        self.reverse_transforms()
        x = y.clone()
        x = reverse_momentum(x)
        x = self.forward(x)
        x = reverse_momentum(x)
        self.reverse_transforms()
        return x

    def inverse_and_ladj(self, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        self.reverse_transforms()
        y = x.clone()
        y = reverse_momentum(y)
        ladj = 0.0
        for transform in self.transforms:
            y, _ladj = transform.inverse_and_ladj(y)
            ladj += _ladj
        y = reverse_momentum(y)
        self.reverse_transforms()
        return (y, ladj)


class LinearTransform(SymplecticTransform):
    def __init__(self, matrix: torch.Tensor) -> None:
        super().__init__()
        self.matrix = matrix

    def set_matrix(self, matrix: torch.Tensor) -> None:
        self.matrix = matrix
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.linear(x, self.matrix)
    
    def to(self, device):
        self.matrix = self.matrix.to(device)
        return self


class QuadrupoleTransform(SymplecticTransform):
    def __init__(self):
        raise NotImplementedError
        

class MultipoleTransform(SymplecticTransform):
    def __init__(self):
        raise NotImplementedError


class ProjectionTransform1D(Transform):
    def __init__(self, vector: torch.Tensor):
        super().__init__()
        self.vector = vector / torch.norm(vector)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sum(x * self.vector, dim=1)[:, None]
        

def rotation_matrix(angle):
    _cos = np.cos(angle)
    _sin = np.sin(angle)
    return torch.tensor([[_cos, _sin], [-_sin, _cos]])


def reverse_momentum(x):
    for i in range(0, x.shape[1], 2):
        x[:, i + 1] *= -1.0
    x = self.forward(x)
    for i in range(0, x.shape[1], 2):
        x[:, i + 1] *= -1.0
    return x
