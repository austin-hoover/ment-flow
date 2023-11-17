"""Phase space distribution analysis tools."""
import numpy as np


def twiss_2x2(Sigma):
    """RMS Twiss parameters from 2 x 2 covariance matrix.

    Parameters
    ----------
    cov : ndaray, shape (2, 2)
        The covariance matrix for position u and momentum u' [[<uu>, <uu'>], [<uu'>, <u'u'>]].

    Returns
    -------
    alpha : float
        The alpha parameter (-<uu'> / sqrt(<uu><u'u'> - <uu'>^2)).
    beta : float
        The beta parameter (<uu> / sqrt(<uu><u'u'> - <uu'>^2)).
    """
    eps = emittance_2x2(cov)
    beta = sigma[0, 0] / eps
    alpha = -sigma[0, 1] / eps
    return alpha, beta


def emittance_2x2(cov):
    """RMS emittance from u-u' covariance matrix.

    Parameters
    ----------
    Sigma : ndaray, shape (2, 2)
        The covariance matrix for position u and momentum u' [[<uu>, <uu'>], [<uu'>, <u'u'>]].

    Returns
    -------
    float
        The RMS emittance (sqrt(<uu><u'u'> - <uu'>^2)).
    """
    return np.sqrt(np.linalg.det(cov))


def apparent_emittance(cov):
    """RMS apparent emittances from d x d covariance matrix.

    Parameters
    ----------
    cov : ndarray, shape (d x d)
        A covariance matrix. (Dimensions ordered {x, x', y, y', ...}.)

    Returns
    -------
    eps_x, eps_y, eps_z, ... : float
        The emittance in each phase-plane (eps_x, eps_y, eps_z, ...)
    """
    emittances = []
    for i in range(0, Sigma.shape[0], 2):
        emittances.append(emittance_2x2(cov[i : i + 2, i : i + 2]))
    if len(emittances) == 1:
        emittances = emittances[0]
    return emittances


def twiss(cov):
    """RMS Twiss parameters from d x d covariance matrix.

    Parameters
    ----------
    cov : ndarray, shape (d, d)
        A covariance matrix. (Dimensions ordered {x, x', y, y', ...}.)

    Returns
    -------
    alpha_x, beta_x, alpha_y, beta_y, alpha_z, beta_z, ... : float
        The Twiss parameters in each plane.
    """
    params = []
    for i in range(0, cov.shape[0], 2):
        params.extend(twiss_2x2(cov[i : i + 2, i : i + 2]))
    return params


def norm_xxp_yyp_zzp(X, scale_emittance=False):
    """Normalize x-px, y-py, z-pz.

    Parameters
    ----------
    X : ndarray, shape (n, d)
        Coordinates of n points in d-dimensional phase space (d is even).
    scale_emittance : bool
        Whether to divide the coordinates by the square root of the rms emittance.

    Returns
    -------
    Xn : ndarray, shape (n, d)
        Normalized phase space coordinate array.
    """
    if X.shape[1] % 2 != 0:
        raise ValueError("X must have an even number of columns.")
    Xn = np.zeros(X.shape)
    cov = np.cov(X.T)
    for i in range(0, X.shape[1], 2):
        _cov = cov[i : i + 2, i : i + 2]
        alpha, beta = twiss(_cov)
        Xn[:, i] = X[:, i] / np.sqrt(beta)
        Xn[:, i + 1] = (np.sqrt(beta) * X[:, i + 1]) + (alpha * X[:, i] / np.sqrt(beta))
        if scale_emittance:
            Xn[:, i : i + 2] = Xn[:, i : i + 2] / apparent_emittance(_cov)
    return Xn