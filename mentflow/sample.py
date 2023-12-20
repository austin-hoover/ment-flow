import math
from typing import Callable
from typing import Iterable
from typing import List
from typing import Tuple

import numpy as np
import torch

from mentflow.utils import grab
from mentflow.utils.image import get_grid_points_torch
from mentflow.utils.image import sample_hist_torch


class GridSampler:
    def __init__(self, limits: Tuple[Tuple[float], Tuple[float]] = None, res: int = 50, device=None) -> None:
        self.res = res
        self.limits = limits
        self.d = len(limits)
        self.initialize(limits=limits, res=res)
        self.device = device
        if self.device is None:
            self.device = torch.device("cpu")

    def _send(self, x):
        return x.type(torch.float32).to(self.device)

    def initialize(self, limits=None, res=None):
        if limits is not None:
            self.limits = limits
        if res is not None:
            self.res = res
        self.d = len(limits)
        self.coords = [
            torch.linspace(self.limits[i][0], self.limits[i][1], self.res) 
            for i in range(self.d)
        ]
        self.shape = self.d * [self.res]

    def get_mesh(self) -> List[torch.Tensor]:
        return torch.meshgrid(*self.coords, indexing="ij")

    def get_grid_points(self) -> torch.Tensor:
        return torch.vstack([C.ravel() for C in self.get_mesh()]).T
        
    def __call__(self, log_prob_func: Callable, n: int) -> torch.Tensor:
        grid_points = self.get_grid_points()
        grid_points = self._send(grid_points)
        log_values = log_prob_func(grid_points)
        values = torch.exp(log_values)
        values = values.reshape(self.shape)
        x = sample_hist_torch(values, self.coords, n)
        x = self._send(x)
        return x
