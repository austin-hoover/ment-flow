import numpy as np

from .dist import Distribution
from .dist import decorrelate


class Gaussian(Distribution):
    def __init__(self, **kws):
        super().__init__(**kws)

    def _sample(self, n):
        return self.rng.normal(size=(n, self.d))
        

class KV(Distribution):
    def __init__(self, **kws):
        super().__init__(**kws)
        if self.noise is None:
            self.noise = 0.05

    def _sample(self, n):
        X = self.rng.normal(size=(n, self.d))
        X = np.apply_along_axis(lambda row: row / np.linalg.norm(row), 1, X)
        X = X / np.std(X, axis=0)
        return X
        

class WaterBag(Distribution):
    def __init__(self, **kws):
        super().__init__(**kws)
        if self.noise is None:
            self.noise = 0.05

    def _sample(self, n):
        X = KV(d=self.d, rng=self.rng).sample_numpy(n)
        r = self.rng.uniform(0.0, 1.0, size=n) ** (1.0 / self.d)
        r = r[:, None]
        X = X * r
        X = X / np.std(X, axis=0)
        return X


class Hollow(Distribution):
    def __init__(self, exp=0.25, **kws):
        super().__init__(**kws)
        self.exp = exp
        if self.noise is None:
            self.noise = 0.05

    def _sample(self, n):
        X = KV(d=self.d, rng=self.rng).sample_numpy(n)
        r = self.rng.uniform(0.0, 1.0, size=X.shape[0]) ** self.exp
        X = X * r[:, None]
        X = X / np.std(X, axis=0)
        return X


class Rings(Distribution):
    def __init__(self, n_rings=2, **kws):
        super().__init__(**kws)
        self.n_rings = n_rings
        if self.noise is None:
            self.noise = 0.15

    def _sample(self, n):
        n_outer = n // self.n_rings
        sizes = [n - (self.n_rings - 1) * n_outer] + (self.n_rings - 1) * [n_outer]
        radii = np.linspace(0.0, 1.0, self.n_rings + 1)[1:]
        X = []
        dist = KV(d=self.d, rng=self.rng)
        for size, radius in zip(sizes, radii):
            X.append(radius * dist.sample(size))
        X = np.vstack(X)
        X = X / np.std(X, axis=0)
        return X
        


DISTRIBUTIONS = {
    "gaussian": Gaussian,
    "kv": KV,
    "hollow": Hollow,
    "rings": Rings,
    "waterbag": WaterBag,
}


def gen_dist(name, **kws):
    return DISTRIBUTIONS[name](**kws)
