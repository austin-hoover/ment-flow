import numpy as np
import torch

import mentflow as mf

from experiments.utils import *


def make_rotation_transforms(theta_min=0.0, theta_max=180.0, n_transforms=6):
    angles = np.linspace(
        np.radians(theta_min),
        np.radians(theta_max),
        n_transforms,
        endpoint=False
    )
    transforms = []
    for angle in angles:
        matrix = mf.transform.rotation_matrix(angle)
        matrix = matrix.type(torch.float32)
        transform = mf.transform.LinearTransform(matrix)
        transforms.append(transform)
    return transforms