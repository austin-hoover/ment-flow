"""Two-dimensional distributions."""
import os
import sys

import numpy as np
import scipy.stats
import skimage as ski

from mentflow.data import radial
from mentflow.data.distribution import Distribution
from mentflow.data.distribution import normalize
from mentflow.data.radial import Gaussian
from mentflow.data.radial import Hollow
from mentflow.data.radial import Rings
from mentflow.sample import sample_hist



class EightGaussians(Distribution):
    def __init__(self, **kws):
        super().__init__(d=2, **kws)
        if self.noise is None:
            self.noise = 0.20

    def _sample(self, n):
        theta = 2.0 * np.pi * self.rng.integers(0, 8, n) / 8.0
        x = np.cos(theta)
        y = np.sin(theta)
        X = np.stack([x, y], axis=-1)
        X *= 1.5
        return X


class Galaxy(Distribution):
    def __init__(self, turns=5, truncate=3.0, **kws):
        super().__init__(d=2, **kws)
        self.turns = turns
        self.truncate = truncate
        if self.noise is None:
            self.noise = 0.0

    def _sample(self, n):
        
        def _rotate(X, theta):
            x = X[:, 0].copy()
            y = X[:, 1].copy()
            X[:, 0] = x * np.cos(theta) + y * np.sin(theta)
            X[:, 1] = y * np.cos(theta) - x * np.sin(theta)
            return X

        # Start with flattened Gaussian distribution.
        X = np.zeros((n, 2))
        X[:, 0] = 1.0 * scipy.stats.truncnorm.rvs(-self.truncate, self.truncate, size=n)
        X[:, 1] = 0.5 * scipy.stats.truncnorm.rvs(-self.truncate, self.truncate, size=n)

        # Apply amplitude-dependent phase advance.
        r = np.linalg.norm(X, axis=1)
        r = r / np.max(r)        
        theta = 2.0 * np.pi * (1.0 + 0.5 * (r **0.25))
        for _ in range(self.turns):
            X = _rotate(X, theta)     

        # Standardize the data set.
        X = X / np.std(X, axis=0)
        X = X * 0.85
        return X


class KV(Distribution):
    """Projection of 4D Kapchinskij-Vladimirskij (microcanonical) distribution."""
    def __init__(self, **kws):
        super().__init__(**kws)
        if self.noise is None:
            self.noise = 0.05

    def _sample(self, n):
        X = radial.KV(d=4, rng=self.rng).sample_numpy(n)
        X = X[:, :2]
        X = X / np.std(X, axis=0)
        return X


class Leaf(Distribution):
    def __init__(self, xmax=2.5, **kws):
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

    def _sample(self, n):
        x = sample_hist(self.hist, self.edges, n=n)
        return x


class Pinwheel(Distribution):
    def __init__(self, **kws):
        super().__init__(d=2, **kws)
        if self.noise is None:
            self.noise = 0.10

    def _sample(self, n):
        theta = 2.0 * np.pi * self.rng.integers(0, 5, n) / 5.0
        a = self.rng.normal(loc=1.0, scale=0.25, size=n)
        b = self.rng.normal(scale=0.1, size=n)
        theta = theta + np.exp(a - 1.0)
        x = a * np.cos(theta) - b * np.sin(theta)
        y = a * np.sin(theta) + b * np.cos(theta)
        X = np.stack([x, y], axis=-1)
        X = X / np.std(X, axis=0)
        return X


class SwissRoll(Distribution):
    def __init__(self, **kws):
        super().__init__(d=2, **kws)
        if self.noise is None:
            self.noise = 0.15

    def _sample(self, n):
        t = 1.5 * np.pi * (1.0 + 2.0 * self.rng.uniform(0.0, 1.0, size=n))
        X = np.stack([t * np.cos(t), t * np.sin(t)], axis=-1)
        X = X / np.std(X, axis=0)
        return X


class TwoSpirals(Distribution):
    def __init__(self, exp=0.65, **kws):
        super().__init__(d=2, **kws)
        self.exp = exp
        if self.noise is None:
            self.noise = 0.070

    def _sample(self, n):
        self.exp = 0.75
        t = 3.0 * np.pi * np.random.uniform(0.0, 1.0, size=n) ** self.exp    
        r = t / 2.0 / np.pi * np.sign(self.rng.normal(size=n))
        t = t + self.rng.normal(size=n, scale=np.linspace(0.0, 1.0, n))
        X = np.stack([-r * np.cos(t), r * np.sin(t)], axis=-1)
        X = X / np.std(X, axis=0)
        return X


class WaterBag(Distribution):
    def __init__(self, **kws):
        super().__init__(**kws)

    def _sample(self, n):
        X = radial.WaterBag(d=4, rng=self.rng).sample_numpy(n)
        X = X[:, :2]
        X = X / np.std(X, axis=0)
        return X



DISTRIBUTIONS = {
    "eight-gaussians": EightGaussians,
    "galaxy": Galaxy,
    "gaussian": Gaussian,
    "hollow": Hollow,
    "kv": KV,
    "leaf": Leaf,
    "pinwheel": Pinwheel,
    "rings": Rings,
    "swissroll": SwissRoll,
    "two-spirals": TwoSpirals,
    "waterbag": WaterBag,
}


def gen_dist(name, **kws):
    return DISTRIBUTIONS[name](**kws)


