import os
import sys
from typing import Callable
from typing import List

import numpy as np
import scipy.stats
import skimage as ski
import torch

from mentflow.sample import sample_hist
from mentflow.utils import sphere_surface_area
from .utils import corrupt
from .utils import decorrelate
from .utils import normalize
from .utils import shuffle


class Distribution:
    def __init__(
        self, ndim: int = 2,
        seed: int = None, 
        normalize: bool = False, 
        shuffle: bool = True, 
        decorr: bool = False, 
        noise: float = None,
        shear: Callable = None,
        device: torch.device = None,
    ) -> None:
        self.ndim = ndim
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.normalize = normalize
        self.shuffle = shuffle
        self.decorr = decorr
        self.noise = noise
        self.shear = shear
        self.device = device

    def _sample(self, size: int) -> np.ndarray:
        raise NotImplementedError

    def _log_prob(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError
        
    def sample_np(self, size: int) -> np.ndarray:
        x = self._sample(int(size))
        if self.shuffle:
            x = shuffle(x, rng=self.rng)
        if self.normalize:
            x = normalize(x)
        if self.noise:
            x = corrupt(x, self.noise, rng=self.rng)
        if self.decorr:
            x = decorrelate(x, rng=self.rng)
        if self.shear:
            sigma_old = np.std(x[:, 0])
            x[:, 0] += self.shear * x[:, 1]
            sigma_new = np.std(x[:, 0])
            x[:, 0] *= (sigma_old / sigma_new)
        return x

    def sample(self, size: int) -> torch.Tensor:
        x = self.sample_np(size)
        x = torch.from_numpy(x)
        x = x.type(torch.float32)
        x = x.to(self.device)
        return x
        
    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        log_prob = self._log_prob(x.detach().cpu().numpy())
        log_prob = torch.from_numpy(log_prob)
        log_prob = log_prob.type(torch.float32)
        log_prob = log_prob.to(device)
        return log_prob


class EightGaussians(Distribution):
    def __init__(self, **kws) -> None:
        kws["ndim"] = 2
        super().__init__(**kws)
        if self.noise is None:
            self.noise = 0.20

    def _sample(self, size: int) -> np.ndarray:
        theta = 2.0 * np.pi * self.rng.integers(0, 8, size) / 8.0
        x = np.stack([np.cos(theta), np.sin(theta)], axis=-1)
        x *= 1.5
        return x


class Galaxy(Distribution):
    def __init__(self, turns: int = 5, truncate: float = 3.0, **kws) -> None:
        kws["ndim"] = 2
        super().__init__(**kws)
        self.turns = turns
        self.truncate = truncate
        if self.noise is None:
            self.noise = 0.0

    def _sample(self, size: int) -> np.ndarray:
        
        def _rotate(x, theta):
            _x = x[:, 0].copy()
            _y = x[:, 1].copy()
            x[:, 0] = _x * np.cos(theta) + _y * np.sin(theta)
            x[:, 1] = _y * np.cos(theta) - _x * np.sin(theta)
            return x

        # Start with flattened Gaussian distribution.
        x = np.zeros((size, 2))
        x[:, 0] = 1.0 * scipy.stats.truncnorm.rvs(-self.truncate, self.truncate, size=size)
        x[:, 1] = 0.5 * scipy.stats.truncnorm.rvs(-self.truncate, self.truncate, size=size)

        # Apply amplitude-dependent phase advance.
        r = np.linalg.norm(x, axis=1)
        r = r / np.max(r)        
        theta = 2.0 * np.pi * (1.0 + 0.5 * (r **0.25))
        for _ in range(self.turns):
            x = _rotate(x, theta)     

        # Normalize
        x /= np.std(x, axis=0)
        x *= 0.85
        return x


class Gaussian(Distribution):
    def __init__(self, **kws) -> None:
        super().__init__(**kws)

    def _sample(self, size: int) -> np.ndarray:
        return self.rng.normal(size=(size, self.ndim))


class GaussianMixture(Distribution):
    def __init__(
        self, 
        modes: int = 7, 
        xmax: float = 3.0, 
        scale: float = 0.75, 
        shiftscale=True, 
        **kws
    ) -> None:
        super().__init__(**kws)
        self.modes = modes
        self.locs = self.rng.uniform(-xmax, xmax, size=(self.modes, self.ndim))
        self.scales = scale * np.ones(self.modes)
        self.shiftscale = shiftscale
        
    def _sample(self, size: int) -> np.ndarray:
        x = [
            self.rng.normal(loc=loc, scale=scale, size=(size // self.modes, self.ndim))
            for scale, loc in zip(self.scales, self.locs)
        ]
        x = np.vstack(x)
        if self.shiftscale:
            x = x - np.mean(x, axis=0)
            x = x / np.std(x, axis=0)
        return x


class Hollow(Distribution):
    def __init__(self, exp: float = 1.66, **kws) -> None:
        super().__init__(**kws)
        self.exp = exp
        if self.noise is None:
            self.noise = 0.05

    def _sample(self, size: int) -> np.ndarray:
        x = KV(ndim=self.ndim, seed=self.seed).sample_np(size)
        r = self.rng.uniform(0.0, 1.0, size=x.shape[0]) ** (1.0 / (self.exp * self.ndim))
        x *= r[:, None]
        x /= np.std(x, axis=0)
        return x


class KV(Distribution):
    def __init__(self, **kws) -> None:
        super().__init__(**kws)
        if self.noise is None:
            self.noise = 0.05

    def _sample(self, size: int) -> np.ndarray:
        x = self.rng.normal(size=(size, self.ndim))
        x /= np.linalg.norm(x, axis=1)[:, None]
        x /= np.std(x, axis=0)
        return x


class Leaf(Distribution):
    def __init__(self, xmax: float = 2.5, **kws) -> None:
        kws["ndim"] = 2
        super().__init__(**kws)
        if self.noise is None:
            self.noise = 0.010
        self.xmax = xmax

        path = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(path, "./files/leaf.png")
        self.hist = ski.io.imread(path, as_gray=True)
        self.hist = 1.0 - self.hist
        self.hist = self.hist[::-1, :].T
        self.edges = [np.linspace(-self.xmax, +self.xmax, s + 1) for s in self.hist.shape]

    def _sample(self, size: int) -> np.ndarray:
        hist = self.hist
        edges = self.edges
        
        pdf = hist.ravel()
        idx = np.flatnonzero(pdf)
        pdf = pdf[idx]
        pdf = pdf / np.sum(pdf)
        idx = np.random.choice(idx, size, replace=True, p=pdf)
        idx = np.unravel_index(idx, shape=hist.shape)
        lb = [edges[axis][idx[axis]] for axis in range(hist.ndim)]
        ub = [edges[axis][idx[axis] + 1] for axis in range(hist.ndim)]
        x = np.random.uniform(lb, ub).T
        return x


class Pinwheel(Distribution):
    def __init__(self, **kws) -> None:
        super().__init__(**kws)
        if self.noise is None:
            self.noise = 0.10

    def _sample(self, size: int) -> np.ndarray:
        a = self.rng.normal(loc=1.0, scale=0.25, size=size)
        b = self.rng.normal(scale=0.1, size=size)
        theta = 2.0 * np.pi * self.rng.integers(0, 5, size) / 5.0
        theta = theta + np.exp(a - 1.0)
        x = np.stack(
            [
                a * np.cos(theta) - b * np.sin(theta),
                a * np.sin(theta) + b * np.cos(theta),
            ],
            axis=-1
        )
        x /= np.std(x, axis=0)
        return x


class Rings(Distribution):
    def __init__(self, n_rings: int = 2, decay: float = 0.5, **kws) -> None:
        super().__init__(**kws)
        self.n_rings = n_rings
        self.decay = decay
        if self.noise is None:
            self.noise = 0.15

    def _sample(self, size: int) -> np.ndarray:        
        # Set sphere radii.
        radii = np.linspace(1.0, 0.0, self.n_rings, endpoint=False)[::-1]
                
        # Set equal particle density on each sphere.
        sizes = np.array([sphere_surface_area(d=self.ndim, r=r) for r in radii])
        
        # Make density vary linearly with the radius.
        sizes = sizes * np.linspace(1.0, self.decay, self.n_rings)

        # Scale to correct total particle number.
        sizes = sizes * (size / np.sum(sizes))
        sizes = sizes.astype(int)
        
        # Generate particles on each sphere.
        x = []
        dist = KV(ndim=self.ndim, seed=self.seed)
        for size, radius in zip(sizes, radii):
            x.append(radius * dist.sample(size))
        x = np.vstack(x)
        x /= np.std(x, axis=0)
        return x


class SwissRoll(Distribution):
    def __init__(self, **kws) -> None:
        super().__init__(**kws)
        if self.noise is None:
            self.noise = 0.15

    def _sample(self, size: int) -> np.ndarray:
        t = 1.5 * np.pi * (1.0 + 2.0 * self.rng.uniform(0.0, 1.0, size=size))
        x = np.stack([t * np.cos(t), t * np.sin(t)], axis=-1)
        x /= np.std(x, axis=0)
        return x


class TwoSpirals(Distribution):
    def __init__(self, exp=0.65, **kws) -> None:
        super().__init__(**kws)
        self.exp = exp
        if self.noise is None:
            self.noise = 0.070

    def _sample(self, size: int) -> np.ndarray:
        self.exp = 0.75
        t = 3.0 * np.pi * np.random.uniform(0.0, 1.0, size=size) ** self.exp    
        r = t / 2.0 / np.pi * np.sign(self.rng.normal(size=size))
        t = t + self.rng.normal(size=size, scale=np.linspace(0.0, 1.0, size))
        x = np.stack([-r * np.cos(t), r * np.sin(t)], axis=-1)
        x = x / np.std(x, axis=0)
        return x


class WaterBag(Distribution):
    def __init__(self, **kws) -> None:
        super().__init__(**kws)
        if self.noise is None:
            self.noise = 0.05

    def _sample(self, size: int) -> np.ndarray:
        x = KV(ndim=self.ndim, seed=self.seed).sample_np(size)
        r = self.rng.uniform(0.0, 1.0, size=x.shape[0]) ** (1.0 / self.ndim)
        x *= r[:, None]
        x /= np.std(x, axis=0)
        return x


DISTRIBUTIONS = {
    "eight-gaussians": EightGaussians,
    "galaxy": Galaxy,
    "gaussian": Gaussian,
    "gaussian_mixture": GaussianMixture,
    "hollow": Hollow,
    "kv": KV,
    "leaf": Leaf,
    "pinwheel": Pinwheel,
    "rings": Rings,
    "swissroll": SwissRoll,
    "two-spirals": TwoSpirals,
    "waterbag": WaterBag,
}


def get_distribution(name: str, **kws) -> Distribution:
    return DISTRIBUTIONS[name](**kws)

