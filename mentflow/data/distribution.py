import numpy as np


class Distribution:
    def __init__(self, d=2, rng=None, normalize=True, shuffle=True, noise=None, decorr=False):
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

    def sample(self, n):
        X = self._sample(n)
        if self.shuffle:
            X = shuffle(X, rng=self.rng)
        if self.normalize:
            X = normalize(X)
        if self.noise:
            X = corrupt(X, self.noise, rng=self.rng)
        if self.decorr:
            X = decorrelate(X, rng=self.rng)
        return X


def corrupt(X, scale, rng=None):
    return X + rng.normal(scale=scale, size=X.shape)


def decorrelate(X, rng=None):
    if X.shape[1] % 2 == 0:
        for i in range(0, d, 2):
            j = 2 * i
            idx = rng.permutation(np.arange(n))
            X[:, j : j + 1] = X[idx, j : j + 1]
    else:
        for i in range(0, d, 1):
            idx = rng.permutation(np.arange(n))
            X[:, j] = X[idx, j]
    return x


def normalize(X):
    X = X - np.mean(X, axis=0)
    X = X / np.max(np.std(X, axis=0))
    return X


def shuffle(X, rng=None):
    return rng.permutation(X)
    
