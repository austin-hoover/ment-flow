import numpy as np
import torch

import mentflow as mf


def add_measurement_noise(measurements, scale=0.0, noise_type="gaussian", device=None):
    assert noise_type in ["gaussian", "uniform"]
    
    if scale == 0.0:
        return measurements

    if device is None:
        device = torch.device("cpu")
        
    for i in range(len(measurements)):
        for j in range(len(measurements[i])):
            measurement = measurements[i][j]

            frac_noise = torch.zeros(measurement.shape[0])
            if noise_type == "uniform":
                frac_noise = scale * torch.rand(measurement.shape[0]) * 2.0
            else:
                frac_noise = scale * torch.randn(measurement.shape[0])
            
            frac_noise = frac_noise.type(torch.float32)
            if device is not None:
                frac_noise = frac_noise.to(device)
                
            measurement = measurement * (1.0 + frac_noise)
            measurement = torch.clamp(measurement, 0.0, None)
            measurements[i][j] = measurement
    return measurements


def make_transforms_rotation(theta_min=0.0, theta_max=180.0, n_transforms=6):
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