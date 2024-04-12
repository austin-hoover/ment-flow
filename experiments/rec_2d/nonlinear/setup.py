import numpy as np
import torch
from omegaconf import DictConfig

import mentflow as mf

from experiments.setup import setup_mentflow_model
from experiments.setup import train_mentflow_model
from experiments.setup import setup_ment_model
from experiments.setup import train_ment_model
from experiments.setup import generate_training_data
from experiments.rec_2d.setup import make_diagnostics
from experiments.rec_2d.setup import make_distribution
from experiments.rec_2d.setup import setup_eval
from experiments.rec_2d.setup import setup_plot


mf.train.plot.set_proplot_rc()


def make_transforms(cfg: DictConfig):
    """Generate rotation matrix transforms for uniformly spaced angles."""
    transforms = []

    ## Constant linear focusing, varying multipole.
    order = cfg.meas.mult_order
    strength_max = +cfg.meas.max_mult_strength
    strength_min = -strength_max
    strengths = np.linspace(strength_min, strength_max, cfg.meas.num)
    
    for strength in strengths:
        multipole = mf.simulate.MultipoleTransform(order=order, strength=strength)
    
        angle = np.radians(cfg.meas.max_angle)
        matrix = mf.simulate.rotation_matrix(angle)
        matrix = matrix.type(torch.float32)
        rotation = mf.simulate.LinearTransform(matrix)
        
        transform = mf.simulate.CompositeTransform(multipole, rotation)
        transforms.append(transform)
        
    return transforms