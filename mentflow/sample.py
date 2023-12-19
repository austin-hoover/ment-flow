from typing import Callable
from typing import Iterable
from typing import List
from typing import Tuple

import numpy as np

from mentflow.utils.image import get_grid_points
from mentflow.utils.image import sample_hist


class GridSampler:
    def __init__(self, limits: Tuple[Tuple[float], Tuple[float]] = None, res: int = 50) -> None:
        self.res = res
        self.limits = limits
        self.d = len(limits)
        self.initialize(limits=limits, res=res)

    def initialize(self, limits=None, res=None):
        if limits is not None:
            self.limits = limits
        if res is not None:
            self.res = res
        self.d = len(limits)
        self.edges = [
            np.linspace(self.limits[i][0], self.limits[i][1], self.res + 1)
            for i in range(self.d)
        ]
        self.coords = [0.5 * (e[:-1] + e[1:]) for e in self.edges]
        self.shape = self.d * [self.res]

    def get_grid_points(self) -> np.ndarray:
        return get_grid_points(self.coords)

    def get_mesh(self) -> List[np.ndarray]:
        return np.meshgrid(*self.coords, indexing="ij")
        
    def __call__(self, func: Callable, n: int) -> np.ndarray:
        grid_points = self.get_grid_points()
        values = func(grid_points)
        values = values.reshape(self.shape)
        x = sample_hist(values, self.edges, n)
        return x
