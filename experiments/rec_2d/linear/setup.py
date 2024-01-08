import numpy as np
import torch
from omegaconf import DictConfig

import mentflow as mf

from experiments.setup import generate_training_data
from experiments.setup import setup_model
from experiments.setup import setup_and_run_trainer
from experiments.rec_2d.setup import make_diagnostic
from experiments.rec_2d.setup import make_dist
from experiments.rec_2d.setup import setup_eval
from experiments.rec_2d.setup import setup_plot


mf.train.plot.set_proplot_rc()


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
        matrix = mf.transform.rotation_matrix(angle)
        matrix = matrix.type(torch.float32)
        transform = mf.transform.LinearTransform(matrix)
        transforms.append(transform)
    return transforms
