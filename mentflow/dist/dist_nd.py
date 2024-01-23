import numpy as np

from .dist import Distribution
from .dist import decorrelate

from ..utils import sphere_surface_area


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
        X = X / np.linalg.norm(X, axis=1)[:, None]
        return X
        

class WaterBag(Distribution):
    def __init__(self, **kws):
        super().__init__(**kws)
        if self.noise is None:
            self.noise = 0.05

    def _sample(self, n):
        X = KV(d=self.d, rng=self.rng).sample_numpy(n)
        r = self.rng.uniform(0.0, 1.0, size=X.shape[0]) ** (1.0 / self.d)
        r = r[:, None]
        X = X * r
        return X


class Hollow(Distribution):
    def __init__(self, exp=1.66, **kws):
        super().__init__(**kws)
        self.exp = exp
        if self.noise is None:
            self.noise = 0.05

    def _sample(self, n):
        X = KV(d=self.d, rng=self.rng).sample_numpy(n)
        r = self.rng.uniform(0.0, 1.0, size=X.shape[0]) ** (1.0 / (self.exp * self.d))
        X = X * r[:, None]
        return X


class Rings(Distribution):
    def __init__(self, n_rings=2, decay=1.0, **kws):
        super().__init__(**kws)
        self.n_rings = n_rings
        self.decay = decay
        if self.noise is None:
            self.noise = 0.15

    def _sample(self, n):
        n_outer = n // self.n_rings
        radii = np.linspace(1.0, 0.0, self.n_rings, endpoint=False)[::-1]
        sizes = np.array([sphere_surface_area(d=self.d, r=r) for r in radii])
        sizes = sizes * np.linspace(1.0, self.decay * 1.0 / self.n_rings, self.n_rings)
        sizes = sizes * (n / np.sum(sizes))
        sizes = sizes.astype(int)

        X = []
        dist = KV(d=self.d, rng=self.rng)
        for size, radius in zip(sizes, radii):
            X.append(radius * dist.sample(size))
        X = np.vstack(X)
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
