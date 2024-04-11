import numpy as np


def corrupt(x: np.ndarray, scale: float, rng=None) -> np.ndarray:
    return x + rng.normal(scale=scale, size=x.shape)


def decorrelate(x: np.ndarray, rng=None) -> np.ndarray:
    if x.shape[1] % 2 == 0:
        for i in range(0, d, 2):
            j = 2 * i
            idx = rng.permutation(np.arange(n))
            x[:, j : j + 1] = x[idx, j : j + 1]
    else:
        for i in range(0, d, 1):
            idx = rng.permutation(np.arange(n))
            x[:, j] = x[idx, j]
    return x


def normalize(x: np.ndarray) -> np.ndarray:
    x = x - np.mean(x, axis=0)
    x = x / np.std(x, axis=0)
    return x


def shuffle(x: np.ndarray, rng=None) -> np.ndarray:
    return rng.permutation(x)