import numpy as np

import mentflow.data.radial as radial
from mentflow.data.utils import decorrelate
from mentflow.data.utils import process


def gen_circles(n, rng=None):
    return gen_rings(n, rng=rng, n_rings=2)


def gen_kv(n, rng=None):
    return radial.gen_kv(n, 4, rng=rng)[:, :2]


def gen_waterbag(n, rng=None):
    return radial.gen_waterbag(n, 4, rng=rng)[:, :2]


def gen_hollow(n, rng=None):
    return radial.gen_hollow(n, 2, rng=rng)


def gen_gaussians(n, rng=None):
    theta = 2.0 * np.pi * rng.integers(0, 8, n) / 8.0
    data = np.stack([np.cos(theta), np.sin(theta)], axis=-1)
    return data


def gen_pinwheel(n, rng=None):
    theta = 2.0 * np.pi * rng.integers(0, 5, n) / 5.0
    a = rng.normal(loc=1.0, scale=0.25, size=n)
    b = rng.normal(scale=0.1, size=n)
    theta = theta + np.exp(a - 1.0)
    x = a * np.cos(theta) - b * np.sin(theta)
    y = a * np.sin(theta) + b * np.cos(theta)
    data = np.stack([x, y], axis=-1)
    return data


def gen_rings(n, rng=None, n_rings=4):
    return radial.gen_rings(n, 2, rng=rng, n_rings=n_rings)


def gen_spirals(n, rng=None, exp=0.65):
    t = 3.0 * np.pi * np.random.uniform(0.0, 1.0, size=n) ** exp    
    r = t / 2.0 / np.pi * np.sign(rng.normal(size=n))
    t = t + rng.normal(size=n, scale=np.linspace(0.0, 1.0, n))
    return np.stack([-r * np.cos(t), r * np.sin(t)], axis=-1)


def gen_swissroll(n, rng=None):
    t = 1.5 * np.pi * (1.0 + 2.0 * rng.uniform(0.0, 1.0, size=n))
    return np.stack([t * np.cos(t), t * np.sin(t)], axis=-1)


def gen_data(
    name="circles",
    size=1000,
    seed=None,
    shuffle=True,
    noise=None,
    decorr=False,
    warp=False,
    warp_kws=None,
    **kws
):
    options = {
        "circles": {
            "func": gen_circles,
            "noise": 0.15,
            "kws": {},
        },
        "gaussians": {
            "func": gen_gaussians,
            "noise": 0.20,
            "kws": {},
        },
        "hollow": {
            "func": gen_hollow,
            "noise": 0.05,
            "kws": {},
        },
        "kv": {
            "func": gen_kv,
            "noise": 0.05,
            "kws": {},
        },
        "pinwheel": {
            "func": gen_pinwheel,
            "noise": 0.10,
            "kws": {},
        },
        "rings": {
            "func": gen_rings,
            "noise": 0.10,
            "kws": {},
        },
        "spirals": {
            "func": gen_spirals,
            "noise": 0.075,
            "kws": {},
        },
        "swissroll": {
            "func": gen_swissroll,
            "noise": 0.15,
            "kws": {},
        },
        "waterbag": {
            "func": gen_waterbag,
            "noise": 0.0,
            "kws": {},
        },
    }

    rng = np.random.default_rng(seed)

    func = options[name]["func"]
    
    for key, val in options[name]["kws"].items():
        kws.setdefault(key, val)

    if noise is None:
        noise = options[name]["noise"]
    
    x = func(size, rng=rng, **kws)
    x = process(x, normalize=True, shuffle=shuffle, noise=noise, rng=rng)
    
    if warp:
        if warp_kws is None:
            warp_kws = dict()
        scale = warp_kws.get("scale", 0.15)
        exp = warp_kws.get("exp", 3.0)
        x[:, 0] += scale * x[:, 1] ** exp
        x[:, 1] -= scale * x[:, 0] ** exp
        x = process(x, normalize=True)
    
    x = process(x, decorr=decorr, rng=rng)
    return x
