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



mf.train.plot.set_proplot_rc()


def make_transforms(cfg: DictConfig):
    transforms = []
    
    if cfg.meas.optics == "corner":
        transforms = []
        axis_meas = (0, 2)
        for i in range(cfg.d):
            for j in range(i):
                matrices = []
                for k, l in zip(axis_meas, (j, i)):
                    matrix = torch.eye(cfg.d)
                    matrix[k, k] = matrix[l, l] = 0.0
                    matrix[k, l] = matrix[l, k] = 1.0
                    matrix = matrix.float()
                    matrices.append(matrix)
                matrix = torch.linalg.multi_dot(matrices[::-1])
                
                transform = mf.sim.LinearTransform(matrix)
                transform = transform.to(cfg.device)
                transforms.append(transform)
                
    return transforms[:-1]
            

def make_diagnostics(cfg: DictConfig) -> List[mf.diag.Diagnostic]:
    device = torch.device(cfg.device)
    
    bin_edges = [
        torch.linspace(-cfg.meas.xmax, cfg.meas.xmax, cfg.meas.bins + 1),
        torch.linspace(-cfg.meas.xmax, cfg.meas.xmax, cfg.meas.bins + 1),
    ]
    diagnostic = mf.diag.Histogram2D(
        axis=(0, 2),
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
        mf.train.plot.PlotProj2D(),
    ]
    plot_dist = [
        mf.train.plot.PlotDistRadialPDF(
            fig_kws=None,
            bins=50,
            rmax=3.5,
            kind="step",
            lw=1.5,
        ),
        mf.train.plot.PlotDistRadialCDF(
            fig_kws=None,
            bins=50,
            rmax=3.5,
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
