"""Setup for n:2 reconstructions."""
from typing import Callable
from typing import List

import numpy as np
import torch
import proplot as pplt
from omegaconf import DictConfig
from omegaconf import OmegaConf

import mentflow as mf
from mentflow.utils import unravel

from experiments.setup import generate_training_data
from experiments.setup import setup_mentflow_model
from experiments.setup import train_mentflow_model
from experiments.setup import setup_ment_model
from experiments.setup import train_ment_model
from experiments.setup import get_discrepancy_function
from experiments.setup import setup_eval


mf.train.plot.set_proplot_rc()


def make_transforms(cfg: DictConfig):
    transforms = []

    # Measure all 2D projections in corner plot.
    if cfg.meas.optics == "corner":
        axis_meas = (0, 2)
        for i in range(cfg.ndim):
            for j in range(i):
                matrices = []
                for k, l in zip(axis_meas, (j, i)):
                    matrix = torch.eye(cfg.ndim)
                    matrix[k, k] = matrix[l, l] = 0.0
                    matrix[k, l] = matrix[l, k] = 1.0
                    matrix = matrix.float()
                    matrices.append(matrix)
                matrix = torch.linalg.multi_dot(matrices[::-1])
                
                transform = mf.simulate.LinearTransform(matrix)
                transform = transform.to(cfg.device)
                transforms.append(transform)

    # Grid scan mux and muy. Valid for 4D reconstructions.
    elif cfg.meas.optics == "phase_scan":
        axis_meas = (0, 2)
        phases_x = torch.linspace(0.0, 2.0 * np.pi, cfg.meas.num)
        phases_y = phases_x
        for mux in phases_x:
            for muy in phases_y:
                matrix = torch.eye(cfg.ndim)
                matrix[0:2, 0:2] = rotation_matrix(mux)
                matrix[2:4, 2:4] = rotation_matrix(muy)
                matrix = matrix.float()
                transform = mf.simulate.LinearTransform(matrix)
                transform = transform.to(cfg.device)
                transforms.append(transform)
                
    return transforms
            

def make_diagnostics(cfg: DictConfig) -> List[mf.diagnostics.Diagnostic]:
    edges = [
        torch.linspace(-cfg.meas.xmax, cfg.meas.xmax, cfg.meas.bins + 1),
        torch.linspace(-cfg.meas.xmax, cfg.meas.xmax, cfg.meas.bins + 1),
    ]
    diagnostic = mf.diagnostics.Histogram2D(
        axis=(0, 2),
        edges=edges,
        bandwidth=(cfg.meas.bandwidth, cfg.meas.bandwidth),
        noise=True,
        noise_scale=cfg.meas.noise_scale, 
        noise_type=cfg.meas.noise_type,
        device=cfg.device,
        seed=cfg.seed,
    )
    diagnostic = diagnostic.to(cfg.device)
    diagnostics = [diagnostic,]
    return diagnostics


def make_distribution(cfg: DictConfig) -> mf.distributions.Distribution:
    """Make distribution from config."""
    kws = OmegaConf.to_container(cfg.dist)
    kws["ndim"] = cfg.ndim
    kws["seed"] = cfg.seed
    kws.pop("size", None)
    dist = mf.distributions.get_distribution(**kws)
    return dist
    

def setup_plot(cfg: DictConfig) -> Callable:
    """Set up plot function from config."""
    plot_proj = [
        mf.train.plot.PlotProj2D(),
    ]
    plot_dist = [
        mf.train.plot.PlotDistRadialSlice2DProj(
            axis_view=(0, 1), 
            slice_radii=np.linspace(3.0, 1.0, 4),
        ),
        
        mf.train.plot.PlotDistCorner(
            bins=85,
            discrete=False, 
            limits=(cfg.ndim * [(-cfg.eval.xmax, +cfg.eval.xmax)]),
            cmaps=[
                pplt.Colormap("mono", right=0.95),
                pplt.Colormap("mono", right=0.95),
            ],
            colors=["black", "black"],
            mask=True,
            diag_kws=dict(kind="line", lw=1.30),
        ),
    ]
    plot = mf.train.plot.PlotModel(
        distribution=make_distribution(cfg), 
        n_samples=cfg.plot.size, 
        plot_proj=plot_proj, 
        plot_dist=plot_dist, 
        device=cfg.device,
    )
    return plot
