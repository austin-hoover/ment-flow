import sys
import numpy as np
import torch
from omegaconf import DictConfig

import mentflow as mf
from mentflow.train.plot import set_proplot_rc

sys.path.append("../../..")
from experiments.setup import setup_mentflow_model
from experiments.setup import train_mentflow_model
from experiments.setup import setup_ment_model
from experiments.setup import train_ment_model
from experiments.setup import generate_training_data
from experiments.rec_2d.setup import make_diagnostics
from experiments.rec_2d.setup import make_distribution
from experiments.rec_2d.setup import setup_eval
from experiments.rec_2d.setup import setup_plot


set_proplot_rc()


def make_transforms(cfg: DictConfig):
    """Generate rotation matrix transforms for uniformly spaced angles."""
    angles = np.linspace(
        np.radians(cfg.meas.min_angle),
        np.radians(cfg.meas.max_angle),
        cfg.meas.num,
        endpoint=False
    )
    transforms = []
    for angle in angles:
        matrix = mf.simulate.rotation_matrix(angle)
        matrix = matrix.type(torch.float32)
        matrix = matrix.to(cfg.device)
        transform = mf.simulate.LinearTransform(matrix)
        transform = transform.to(cfg.device)
        transforms.append(transform)
    return transforms
