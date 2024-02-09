import numpy as np

from .dist import Distribution
from .dist import decorrelate

from ..utils import sphere_surface_area


class Gaussian(Distribution):
    def __init__(self, **kws):
        super().__init__(**kws)

    def _sample(self, n):
        return self.rng.normal(size=(n, self.d))


class GaussianMixture(Distribution):
    def __init__(self, modes=7, xmax=3.0, scale=0.75, shiftscale=True, **kws):
        super().__init__(**kws)
        self.modes = modes
        self.locs = self.rng.uniform(-xmax, xmax, size=(self.modes, self.d))
        self.scales = scale * np.ones(self.modes)
        self.shiftscale = shiftscale

    def _prob(self, x: np.ndarray) -> np.ndarray:
        # Does not account for normalization that can occur in `_sample`.
        prob = np.zeros(x.shape[0])
        for scale, loc in zip(self.scales, self.locs):
            prob += np.exp(-0.5 * np.sum(((x - loc) / scale)**2, axis=1))
        return prob / self.modes

    def _log_prob(self, x: np.ndarray) -> np.ndarray:
        return np.log(self._prob(x))

    def _sample(self, n: int) -> np.ndarray:
        x = [
            self.rng.normal(loc=loc, scale=scale, size=(n // self.modes, self.d))
            for scale, loc in zip(self.scales, self.locs)
        ]
        x = np.vstack(x)
        if self.shiftscale:
            # This will be inconsistent with _prob method.
            x = x - np.mean(x, axis=0)
            x = x / np.std(x, axis=0)
        return x
        

class KV(Distribution):
    def __init__(self, **kws):
        super().__init__(**kws)
        if self.noise is None:
            self.noise = 0.05

    def _sample(self, n):
        X = self.rng.normal(size=(n, self.d))
        X = X / np.linalg.norm(X, axis=1)[:, None]
        X = X / np.std(X, axis=0)
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
        X = X / np.std(X, axis=0)
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
        X = X / np.std(X, axis=0)
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
        X = X / np.std(X, axis=0)
        return X
        


DISTRIBUTIONS = {
    "gaussian": Gaussian,
    "gaussian_mixture": GaussianMixture,
    "kv": KV,
    "hollow": Hollow,
    "rings": Rings,
    "waterbag": WaterBag,
}


def gen_dist(name, **kws):
    return DISTRIBUTIONS[name](**kws)
