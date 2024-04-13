"""Setup for n:1 reconstructions."""
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


def make_transforms(cfg: DictConfig) -> List[Callable]:    
    rng = torch.Generator(device=cfg.device)
    if cfg.seed is not None:
        rng.manual_seed(cfg.seed)

    if cfg.meas.optics == "isotropic":

        shape = (cfg.meas.num, cfg.ndim)
        directions = torch.randn(shape, generator=rng, device=cfg.device)
        directions = directions / torch.norm(directions, dim=1)[:, None]
        
        transforms = []
        for direction in directions:
            # We only care about the first dimension.
            M = torch.eye(cfg.ndim)
            M[0, :] = direction
            M = M.float().to(cfg.device)
            
            transform = mf.simulate.LinearTransform(M)
            transform = transform.to(cfg.device)
            transforms.append(transform)
        return transforms

    else:
        raise ValueError(f"Invalid cfg.meas.optics '{cfg.meas.optics}'")


def make_diagnostics(cfg: DictConfig) -> List[mf.diagnostics.Diagnostic]:    
    edges = torch.linspace(-cfg.meas.xmax, cfg.meas.xmax, cfg.meas.bins + 1)
    diagnostic = mf.diagnostics.Histogram1D(
        axis=0,
        edges=edges, 
        bandwidth=cfg.meas.bandwidth,
        noise=True,
        noise_scale=cfg.meas.noise_scale, 
        noise_type=cfg.meas.noise_type,
        device=cfg.device,
        seed=cfg.seed,
    )
    diagnostic = diagnostic.to(cfg.device)
    diagnostics = [diagnostic,]  # one diagnostic per transform
    return diagnostics


def make_distribution(cfg: DictConfig) -> mf.distributions.Distribution:
    kws = OmegaConf.to_container(cfg.dist)
    kws["ndim"] = cfg.ndim
    kws["seed"] = cfg.seed
    kws.pop("size", None)
    distribution = mf.distributions.get_distribution(**kws)
    return distribution
    

def setup_plot(cfg: DictConfig) -> Callable:
    """Set up plot function from config."""
    plot_proj = [
        mf.train.plot.PlotProj1D(
            kind="line",
            maxcols=7,
        ),
    ]
    plot_dist = [
        # mf.train.plot.PlotDistRadialPDF(
        #     fig_kws=None,
        #     bins=50,
        #     rmax=3.5,
        #     kind="step",
        #     lw=1.5,
        # ),
        # mf.train.plot.PlotDistRadialCDF(
        #     fig_kws=None,
        #     bins=50,
        #     rmax=3.5,
        #     kind="step",
        #     lw=1.5,
        # ),

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
