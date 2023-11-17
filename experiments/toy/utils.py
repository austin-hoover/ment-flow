import numpy as np
import scipy.interpolate
import skimage.transform
import torch

from mentflow.utils import get_grid_points


def reconstruct_sart(radon_image, angles, iterations=10):
    image = None
    for _ in range(iterations):
        image = skimage.transform.iradon_sart(
            radon_image, theta=np.degrees(angles), image=image, clip=(0.0, None)
        )
    return image.T


def reconstruct_fbp(radon_image, angles):
    image = skimage.transform.iradon(radon_image, theta=np.degrees(angles))
    image = np.clip(image, 0.0, None)
    return image.T


def reconstruct(measurements, angles, method="sart", iterations=10):
    radon_image = torch.vstack(measurements).T
    radon_image = radon_image.detach().cpu().numpy()
    if method == "sart":
        return reconstruct_sart(radon_image, angles, iterations=iterations)
    elif method == "fbp":
        return reconstruct_fbp(radon_image, angles)
    else:
        raise ValueError()
