"""Setup for 2:1 reconstructions."""
from typing import Callable
from typing import List

import numpy as np
import ot
import torch
from omegaconf import DictConfig
from omegaconf import OmegaConf

import mentflow as mf
from mentflow.utils import unravel

from experiments.setup import get_discrepancy_function


def make_diagnostics(cfg: DictConfig) -> List[mf.diagnostics.Diagnostic]:
    """Make one-dimensional histogram diagnostic."""
    device = torch.device(cfg.device)
    
    edges = torch.linspace(-cfg.meas.xmax, cfg.meas.xmax, cfg.meas.bins + 1)
    edges = edges.type(torch.float32)
    edges = edges.to(device)
    
    diagnostic = mf.diagnostics.Histogram1D(
        axis=0, 
        edges=edges, 
        kde=True,
        bandwidth=cfg.meas.bandwidth,
        noise=True,
        noise_scale=cfg.meas.noise_scale, 
        noise_type=cfg.meas.noise_type,
        device=cfg.device,
        seed=cfg.seed,
    )
    diagnostic = diagnostic.to(device)
    diagnostics = [diagnostic,]  # one measurement per transform
    return diagnostics


def make_distribution(cfg: DictConfig) -> mf.distributions.Distribution:
    """Make two-dimensional synthetic distribution from config."""
    kws = OmegaConf.to_container(cfg.dist)
    kws["seed"] = cfg.seed
    kws.pop("size", None)
    dist = mf.distributions.get_distribution(**kws)
    return dist
    

def setup_plot(cfg: DictConfig) -> Callable:
    """Set up plot function from config."""
    plot_proj = mf.train.plot.PlotProj1D(maxcols=7, kind=cfg.plot.line_kind)
    plot_dist = mf.train.plot.PlotDist2D(
        fig_kws=None,
        bins=cfg.plot.bins,
        limits=(2 * [(-cfg.eval.xmax, +cfg.eval.xmax)])
    )
    plot = mf.train.plot.PlotModel(
        distribution=make_distribution(cfg), 
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
        predictions = mf.simulate.forward(x_pred, model.transforms, model.diagnostics)    

        discrepancy_function = get_discrepancy_function(cfg.eval.discrepancy)

        discrepancy_vector = []
        for y_pred, y_meas in zip(unravel(predictions), unravel(model.measurements)):
            discrepancy = discrepancy_function(y_pred, y_meas)
            discrepancy_vector.append(discrepancy)
        discrepancy = sum(discrepancy_vector) / len(discrepancy_vector)

        # Compute distance between true/predicted samples.   
        distance_function = None
        if cfg.eval.distance == "swd":
            distance_function = mf.loss.SlicedWassersteindDistance(n_projections=50, p=2, device=device)

        distance = None
        if distance_function is not None:
            distribution = make_distribution(cfg)
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

