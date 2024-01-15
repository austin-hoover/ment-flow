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


class Transform(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def inverse(self, u: torch.Tensor) -> torch.Tensor:
        return reverse_momentum(self.forward(reverse_momentum(u)))


class CompositeTransform(nn.Module):
    def __init__(self, *transforms) -> None:
        super().__init__()
        self.transforms = nn.Sequential(*transforms)

    def reverse_order(self) -> None:
        self.transforms = nn.Sequential(*reversed([layer for layer in self.transforms]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x
        for transform in self.transforms:
            u = transform(u)
        return u

    def inverse(self, u: torch.Tensor) -> torch.Tensor:
        self.reverse_order()
        x = reverse_momentum(self.forward(reverse_momentum(u)))
        self.reverse_order()
        return x
    
    def to(self, device):
        for transform in self.transforms:
            transform.to(device)
        return self
        

class LinearTransform(Transform):
    def __init__(self, matrix: torch.Tensor) -> None:
        super().__init__()
        self.matrix = matrix

    def set_matrix(self, matrix: torch.Tensor) -> None:
        self.matrix = matrix
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.matmul(x, self.matrix.T)
    
    def to(self, device):
        self.matrix = self.matrix.to(device)
        return self


class OffsetTransform(Transform):
    def __init__(self, delta: float) -> None:
        super().__init__()
        self.delta = delta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x
        for i in range(0, u.shape[1], 2):
            u[:, i] = u[:, i] + self.delta
        return u


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

        ## MPS does not support complex.        
        # z = torch.complex(x, y)
        # zn = torch.pow(z, float(self.order - 1))
        # zn = z ** (self.order - 1)
        # zn_imag = zn.imag
        # zn_real = zn.real

        ## Hard-code a few values of n for now.
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
        

class ProjectionTransform1D(Transform):
    """Computes one-dimensional projection."""
    def __init__(self, vector: torch.Tensor):
        super().__init__()
        self.vector = vector / torch.norm(vector)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sum(x * self.vector, dim=1)[:, None]

    def inverse(self, u: torch.Tensor) -> torch.Tensor:
        raise ValueError("ProjectionTransform1D is not invertible.")
        

def rotation_matrix(angle):
    """Return two-dimensional rotation matrix (angle in radians)."""
    _cos = np.cos(angle)
    _sin = np.sin(angle)
    return torch.tensor([[_cos, _sin], [-_sin, _cos]])


def reverse_momentum(x):
    """Reverse particle momenta. Assume order [x, px, y, py, ...]"""
    for i in range(0, x.shape[1], 2):
        x[:, i + 1] *= -1.0
    return x
