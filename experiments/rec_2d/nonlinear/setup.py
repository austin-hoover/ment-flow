import numpy as np
import torch
from omegaconf import DictConfig

import mentflow as mf

from experiments.setup import setup_mentflow_model
from experiments.setup import train_mentflow_model
from experiments.setup import setup_ment_model
from experiments.setup import train_ment_model
from experiments.setup import generate_training_data
from experiments.rec_2d.setup import make_diagnostic
from experiments.rec_2d.setup import make_dist
from experiments.rec_2d.setup import setup_eval
from experiments.rec_2d.setup import setup_plot


mf.train.plot.set_proplot_rc()


def make_transforms(cfg: DictConfig):
    transforms = []

    ## Constant linear focusing, varying multipole.
    order = 4
    strength_max = +1.0
    strength_min = -strength_max
    strengths = np.linspace(strength_min, strength_max, cfg.meas.num)

    angles = np.radians(np.linspace(0.0, 180.0, cfg.meas.num, endpoint=False))
    
    for strength, angle in zip(strengths, angles):
        multipole = mf.sim.MultipoleTransform(order=order, strength=strength)
    
        matrix = mf.sim.rotation_matrix(angle)
        matrix = matrix.type(torch.float32)
        transform = mf.sim.LinearTransform(matrix)
        
        transform = mf.sim.CompositeTransform(multipole, transform)
        
        transforms.append(transform)

    return transforms