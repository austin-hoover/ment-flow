import numpy as np
import scipy.interpolate
import skimage.transform
import torch

import mentflow as mf


def reconstruct_tomo(measurements, angles, method="sart", iterations=10):
    """Reconstruct image from linear projections using standard tomography algorithms.

    Parameters
    ----------
    measurements : ndarray, shape (k, m)
        List of k projections, each with m bins.
    angles : ndarray, shape (k,)
        List of projection angles.
    method : str
        Tomography method. Options: {"fbp", "sart"}.
    iterations : int
        Number of iterations if method is iterative.

    Returns
    -------
    image : ndarray, shape (m, m)
        The reconstructed distribution.
    """
    radon_image = np.vstack(measurements).T
    angles = np.degrees(-angles)
    
    image = None
    if method == "fbp":
        image = skimage.transform.iradon(radon_image, theta=angles)
    elif method == "sart":
        for _ in range(iterations):
            image = skimage.transform.iradon_sart(
                radon_image, theta=angles, image=image, clip=(0.0, None)
            )
    else:
        raise ValueError("Invalid method name.")
        
    image = np.clip(image, 0.0, None)
    image = image.T
    return image

    