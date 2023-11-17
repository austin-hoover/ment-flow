import numpy as np



def decorrelate(x):
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


def process(x, normalize=False, shuffle=False, noise=0.0, decorr=False, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    if shuffle:
        x = rng.permutation(x)
        
    if normalize:
        x = x - np.mean(x, axis=0)
        x = x / np.max(np.std(x, axis=0))
                
    if noise:
        x = x + rng.normal(scale=noise, size=x.shape)
        
    if decorr:
        x = decorrelate(x)

    return x
