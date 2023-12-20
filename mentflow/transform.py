"""Differentiable transformations.

See the bmad-x repository for beam physics modeling capabilities.
(https://github.com/bmad-sim/Bmad-X)
"""
import numpy as np
import torch
import torch.nn as nn


class CompositeTransform(nn.Module):
    def __init__(self, transforms) -> None:
        super().__init__()
        self.transforms = nn.Sequential(transforms)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for transform in self.transforms:
            x = transform(x)
        return x        

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        for transform in self.transforms:
            x = transform.inverse(x)
        return x

    def to(self, device):
        for transform in self.transforms:
            transform.to(device)
        return self


class Transform(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


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

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.linear(x, self.matrix_inv)

    def to(self, device):
        self.matrix = self.matrix.to(device)
        self.matrix_inv = self.matrix_inv.to(device)
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
