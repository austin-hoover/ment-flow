import numpy as np
import torch

import mentflow.types_


class Distribution:
    def __init__(self, d=2, seed=None, rng=None, normalize=False, shuffle=True, noise=None, decorr=False, x=None):
        self.d = d
        self.rng = rng
        if self.rng is None:
            self.rng = np.random.default_rng(seed)
        self.normalize = normalize
        self.noise = noise
        self.shuffle = shuffle
        self.decorr = decorr
        self.x = x
        if self.x is not None:
            self.d = x.shape[1]

    def _sample(self, n) -> np.ndarray:
        raise NotImplementedError

    def _log_prob(self, x) -> np.ndarray:
        raise NotImplementedError
        
    def sample_numpy(self, n: int) -> np.ndarray:
        x = None
        n = int(n)
        if self.x is not None:
            x = self.x[:n, :]
        else:
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

    def sample(self, n: int) -> torch.Tensor:
        x = self.sample_numpy(n)
        x = torch.from_numpy(x)
        x = x.type(torch.float32)
        return x
        
    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        log_prob = self._log_prob(x.detach().cpu().numpy())
        log_prob = torch.from_numpy(log_prob)
        log_prob = log_prob.type(torch.float32)
        return log_prob


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
    
