import numpy as np

from mentflow.data.utils import decorrelate
from mentflow.data.utils import process


def normalize(x):
    return np.apply_along_axis(lambda row: row / np.linalg.norm(row), 1, x)


def gen_kv(n, d, rng=None):
    return normalize(rng.normal(size=(n, d)))


def gen_waterbag(n, d, rng=None):
    x = gen_kv(n, d, rng=rng)
    r = rng.uniform(0.0, 1.0, size=n) ** (1.0 / d)
    r = r[:, None]
    return x * r


def gen_hollow(n, d, rng=None, exp=0.25):
    x = gen_kv(n, d, rng=rng)
    r = rng.uniform(0.0, 1.0, size=x.shape[0]) ** exp
    return x * r[:, None]


def gen_rings(n, d, rng=None, n_rings=2):
    n_outer = n // n_rings
    sizes = [n - (n_rings - 1) * n_outer] + (n_rings - 1) * [n_outer]
    radii = np.linspace(0.0, 1.0, n_rings + 1)[1:]
    data = []
    for size, radius in zip(sizes, radii):
        x = radius * gen_kv(size, d, rng=rng)
        data.append(x)
    x = np.vstack(data)
    return x


def gen_data(name="waterbag", size=1000, d=2, seed=None, normalize=True, shuffle=True, noise=None, decorr=False, **kws):
    options = {
        "kv": {
            "func": gen_kv,
            "noise": 0.05,
            "kws": {},
        },
        "waterbag": {
            "func": gen_waterbag,
            "noise": 0.05,
            "kws": {},
        },
        "hollow": {
            "func": gen_hollow,
            "noise": 0.05,
            "kws": {},
        },
        "rings": {
            "func": gen_rings,
            "noise": 0.15,
            "kws": {"n_rings": 4},
        },
    }
    rng = np.random.default_rng(seed)
    func = options[name]["func"]
    if noise is None:
        noise = options[name]["noise"]
        
    x = func(size, d, rng=rng, **kws)
    x = process(x, normalize=normalize, shuffle=shuffle, noise=noise, decorr=decorr, rng=rng)
    return x

