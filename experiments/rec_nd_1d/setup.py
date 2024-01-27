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


mf.train.plot.set_proplot_rc()


def make_transforms(cfg: DictConfig):
    """Generate rotation matrix transforms for uniformly spaced angles."""
    device = torch.device(cfg.device)
    
    generator = torch.Generator(device=device)
    if cfg.seed is not None:
        generator.manual_seed(cfg.seed)

    directions = torch.randn((cfg.meas.num, cfg.d), generator=generator, device=device)
    directions = directions / torch.norm(directions, dim=1)[:, None]
    
    transforms = []
    for direction in directions:
        transform = mf.sim.ProjectionTransform(direction)
        transforms.append(transform)
    return transforms


def make_diagnostics(cfg: DictConfig) -> List[mf.diag.Diagnostic]:
    """Make one-dimensional histogram diagnostic."""
    device = torch.device(cfg.device)
    
    bin_edges = torch.linspace(-cfg.meas.xmax, cfg.meas.xmax, cfg.meas.bins + 1)
    bin_edges = bin_edges.type(torch.float32)
    bin_edges = bin_edges.to(device)

    diagnostic = mf.diag.Histogram1D(
        axis=0,
        bin_edges=bin_edges, 
        noise_scale=cfg.meas.noise_scale, 
        noise_type=cfg.meas.noise_type,
        bandwidth=cfg.meas.bandwidth,
        device=cfg.device,
        seed=cfg.seed,
    )
    diagnostic = diagnostic.to(device)
    diagnostics = [diagnostic,]
    return diagnostics


def make_dist(cfg: DictConfig) -> mf.dist.Distribution:
    """Make n-dimensional distribution from config."""
    kws = OmegaConf.to_container(cfg.dist)
    kws["d"] = cfg.d
    kws["seed"] = cfg.seed
    kws.pop("size", None)
    dist = mf.dist.dist_nd.gen_dist(**kws)
    return dist
    

def setup_plot(cfg: DictConfig) -> Callable:
    """Set up plot function from config."""
    plot_proj = [
        mf.train.plot.PlotProj1D(
            kind="line",
            maxcols=7,
        ),
    ]
    plot_dist = [
        mf.train.plot.PlotDistRadialCDF(
            fig_kws=None,
            bins=65,
            rmax=3.0,
            kind="step",
            lw=1.5,
        ),
        mf.train.plot.PlotDistCorner(
            bins=64,
            discrete=False, 
            limits=(cfg.d * [(-cfg.eval.xmax, +cfg.eval.xmax)]),
            cmaps=[
                pplt.Colormap("blues"),
                pplt.Colormap("reds")
            ],
            colors=["blue6", "red6"],
        ),
    ]
    plot = mf.train.plot.PlotModel(
        dist=make_dist(cfg), 
        n_samples=cfg.plot.size, 
        plot_proj=plot_proj, 
        plot_dist=plot_dist, 
        device=cfg.device,
    )
    return plot


def setup_eval(cfg: DictConfig) -> Callable:
    """Set up eval function from config."""
    
    def eval(model): 
        device = cfg.device
        
        # Compute distance between measured/simulated projections.
        x_pred = model.sample(cfg.eval.size)
        x_pred = x_pred.type(torch.float32)
        x_pred = x_pred.to(device)
        predictions = mf.sim.forward(x_pred, model.transforms, model.diagnostics)    

        discrepancy_function = mf.loss.get_loss_function(cfg.eval.disc)

        discrepancy_vector = []
        for y_pred, y_meas in zip(unravel(predictions), unravel(model.measurements)):
            discrepancy = discrepancy_function(y_pred, y_meas)
            discrepancy_vector.append(discrepancy)
        discrepancy = sum(discrepancy_vector) / len(discrepancy_vector)

        # Compute distance between true/predicted samples.   
        distance_function = None
        if cfg.eval.dist == "swd":
            distance_function = mf.loss.SlicedWassersteindDistance(n_projections=50, p=2, device=device)

        distance = None
        if distance_function is not None:
            distribution = make_dist(cfg)
            n_samples = cfg.eval.size
            x_true = distribution.sample(n_samples)    
            x_true = x_true.type(torch.float32)
            x_true = x_true.to(device)
            distance = distance_function(x_pred[:n_samples, :], x_true[:n_samples, :])
            
        # Print summary
        print("")
        print("disc(y_model, y_true) = {}".format(discrepancy))
        if distance is not None:
            print("dist(x_model, x_true) = {}".format(distance))
        
        results = {"discrepancy": discrepancy, "distance": distance}
        return results

    return eval
