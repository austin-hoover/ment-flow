"""Differentiable transformations.

See the bmad-x repository for particle accelerator modeling capabilities (https://github.com/bmad-sim/Bmad-X).
"""
import math
import typing
import weakref
from typing import List
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn


def rotation_matrix(angle):
    _cos = np.cos(angle)
    _sin = np.sin(angle)
    return torch.tensor([[_cos, _sin], [-_sin, _cos]])


def reverse_momentum(x):
    for i in range(0, x.shape[1], 2):
        x[:, i + 1] *= -1.0
    return x


class Transform(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def inverse(self, u: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class CompositeTransform(Transform):
    def __init__(self, *transforms) -> None:
        super().__init__()
        self.transforms = nn.Sequential(*transforms)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x
        for transform in self.transforms:
            u = transform(u)
        return u

    def inverse(self, u: torch.Tensor) -> torch.Tensor:
        x = u
        for transform in self.transforms[::-1]:
            x = transform.inverse(x)
        return x
    
    def to(self, device):
        for transform in self.transforms:
            transform.to(device)
        return self
        

class LinearTransform(Transform):
    def __init__(self, matrix: torch.Tensor) -> None:
        super().__init__()
        self.set_matrix(matrix)

    def set_matrix(self, matrix: torch.Tensor) -> None:
        self.matrix = matrix
        self.matrix_inv = torch.linalg.inv(matrix)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.matmul(x, self.matrix.T)

    def inverse(self, u: torch.Tensor) -> torch.Tensor:
        return torch.matmul(u, self.matrix_inv.T)
    
    def to(self, device):
        self.matrix = self.matrix.to(device)
        return self


class MultipoleTransform(Transform):
    """Apply multipole kick.
    
    https://github.com/PyORBIT-Collaboration/PyORBIT3/blob/main/src/teapot/teapotbase.cc    
    
    Parameters
    ----------
    order: int
        The multipole number (0 for dipole, 1 for quad,, 2 for sextupole, etc.).
    strength : float
        Integrated kick strength [m^(-pole)].
    skew : bool
        If True, rotate the magnet 45 degrees.
    """
    def __init__(self, order: int, strength: float, skew: bool = False) -> None:
        super().__init__()
        self.order = order
        self.strength = strength
        self.skew = skew
        
    def forward(self, X):

        U = X.clone()
        
        x = X[:, 0]        
        if X.shape[1] > 2:
            y = X[:, 2]
        else:
            y = 0.0 * X[:, 0]

        ## MPS does not support torch.complex.        
        # z = torch.complex(x, y)
        # zn = torch.pow(z, float(self.order - 1))
        # zn = z ** (self.order - 1)
        # zn_imag = zn.imag
        # zn_real = zn.real

        ## Hard-code a few values of n.
        if self.order == 1:
            zn_real = 1.0
            zn_imag = 1.0
        if self.order == 2:
            zn_real = x
            zn_imag = y
        if self.order == 3:
            zn_real = (x**2) - (y**2)
            zn_imag = (2.0 * x * y)
        elif self.order == 4:
            zn_real = (+x**3) - (3.0 * y**2 * x)
            zn_imag = (-y**3) + (3.0 * x**2 * y)
        elif self.order == 5:
            zn_real = (x**4) - (6.0 * x**2 * y**2) + (y**4)
            zn_imag = (4.0 * x**3 * y) - (4.0 * x * y**3)
        else:
            raise ValueError("MPS-compatible MultipoleTransform requires order <= 5.")

        k = self.strength / np.math.factorial(self.order - 1)
        if self.skew:
            U[:, 1] = X[:, 1] + k * zn_imag
            if X.shape[1] > 2:
                U[:, 3] = X[:, 3] + k * zn_real
        else:
            U[:, 1] = X[:, 1] - k * zn_real
            if X.shape[1] > 2:
                U[:, 3] = X[:, 1] + k * zn_imag
        return U

    def inverse(self, u: torch.Tensor) -> torch.Tensor:
        return reverse_momentum(self.forward(reverse_momentum(u)))


class ProjectionTransform(Transform):
    def __init__(self, direction: torch.Tensor):
        super().__init__()
        self.direction = direction / torch.norm(direction)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sum(x * self.direction, dim=1)[:, None]
        
