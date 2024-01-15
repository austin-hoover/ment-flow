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
    strength_max = +0.5
    strength_min = -strength_max
    order = 3
    strengths = np.linspace(strength_min, strength_max, cfg.meas.num)
    for strength in strengths:
        multipole = mf.sim.MultipoleTransform(order=order, strength=strength)
    
        angle = np.radians(45.0)
        matrix = mf.sim.rotation_matrix(angle)
        matrix = matrix.type(torch.float32)
        rotation = mf.sim.LinearTransform(matrix)
        
        transform = mf.sim.CompositeTransform(multipole, rotation)
        transforms.append(transform)

    return transforms