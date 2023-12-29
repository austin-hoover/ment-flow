import numpy as np
import torch


class Distribution:
    def __init__(self, d=2, rng=None, normalize=False, shuffle=True, noise=None, decorr=False):
        self.d = d
        self.rng = rng
        if self.rng is None:
            self.rng = np.random.default_rng()
        self.normalize = normalize
        self.noise = noise
        self.shuffle = shuffle
        self.decorr = decorr

    def _sample(self, n):
        raise NotImplementedError

    def prob(self, x):
        raise NotImplementedError 

    def sample_numpy(self, n):
        x = self._sample(n)
        if self.shuffle:
            x = shuffle(x, rng=self.rng)
        if self.normalize:
            x = normalize(x)
        if self.noise:
            x = corrupt(x, self.noise, rng=self.rng)
        if self.decorr:
            x = decorrelate(x, rng=self.rng)
        return x

    def sample(self, n):
        return torch.from_numpy(self.sample_numpy(n))


def corrupt(x, scale, rng=None):
    return x + rng.normal(scale=scale, size=x.shape)


def decorrelate(x, rng=None):
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


def normalize(x):
    x = x - np.mean(x, axis=0)
    x = x / np.std(x, axis=0)
    return x


def shuffle(x, rng=None):
    return rng.permutation(x)
    
