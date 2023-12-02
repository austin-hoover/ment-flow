"""Two-dimensional distributions."""
import numpy as np

from mentflow.data import radial
from mentflow.data.distribution import Distribution
from mentflow.data.distribution import normalize
from mentflow.data.radial import Gaussian
from mentflow.data.radial import Hollow
from mentflow.data.radial import Spheres


class KV(Distribution):
    def __init__(self, **kws):
        super().__init__(**kws)
        if self.noise is None:
            self.noise = 0.05

    def _sample(self, n):
        X = radial.KV(d=4, rng=self.rng).sample(n)
        return X[:, :2]


class Gaussians(Distribution):
    def __init__(self, **kws):
        super().__init__(d=2, **kws)
        if self.noise is None:
            self.noise = 0.20

    def _sample(self, n):
        theta = 2.0 * np.pi * self.rng.integers(0, 8, n) / 8.0
        X = np.stack([np.cos(theta), np.sin(theta)], axis=-1)
        return X


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
        return X


class Spirals(Distribution):
    def __init__(self, exp=0.65, **kws):
        super().__init__(d=2, **kws)
        self.exp = exp
        if self.noise is None:
            self.noise = 0.075

    def _sample(self, n):
        t = 3.0 * np.pi * np.random.uniform(0.0, 1.0, size=n) ** self.exp    
        r = t / 2.0 / np.pi * np.sign(self.rng.normal(size=n))
        t = t + self.rng.normal(size=n, scale=np.linspace(0.0, 1.0, n))
        return np.stack([-r * np.cos(t), r * np.sin(t)], axis=-1)


class SwissRoll(Distribution):
    def __init__(self, **kws):
        super().__init__(d=2, **kws)
        if self.noise is None:
            self.noise = 0.15

    def _sample(self, n):
        t = 1.5 * np.pi * (1.0 + 2.0 * self.rng.uniform(0.0, 1.0, size=n))
        return np.stack([t * np.cos(t), t * np.sin(t)], axis=-1)


class WaterBag(Distribution):
    def __init__(self, **kws):
        super().__init__(**kws)

    def _sample(self, n):
        X = radial.WaterBag(d=4, rng=self.rng).sample(n)
        return X[:, :2]


DISTRIBUTIONS = {
    "circles": Spheres,
    "gaussian": Gaussian,
    "gaussians": Gaussians,
    "hollow": Hollow,
    "kv": KV,
    "pinwheel": Pinwheel,
    "spirals": Spirals,
    "swissroll": SwissRoll,
    "Waterbag": WaterBag,
}


def gen_dist(name, **kws):
    return DISTRIBUTIONS[name](**kws)


