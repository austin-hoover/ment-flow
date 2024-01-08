"""Setup for 2:1 reconstructions."""
from typing import Callable

import numpy as np
import torch
from omegaconf import DictConfig

import mentflow as mf


def make_diagnostic(cfg: DictConfig):
    """Make one-dimensional histogram diagnostic."""
    device = torch.device(cfg.device)
    
    bin_edges = torch.linspace(-cfg.meas.xmax, cfg.meas.xmax, cfg.meas.bins + 1)
    bin_edges = bin_edges.type(torch.float32)
    bin_edges = bin_edges.to(device)
    
    diagnostic = mf.diagnostics.Histogram1D(
        axis=0, 
        bin_edges=bin_edges, 
        noise_scale=cfg.meas.noise_scale, 
        noise_type=cfg.meas.noise_type
    )
    diagnostic = diagnostic.to(device)
    return diagnostic


def make_dist(cfg: DictConfig):
    """Make two-dimensional synthetic distribution from config."""
    dist = mf.data.toy.gen_dist(
        name=cfg.data.name,
        noise=cfg.data.noise,
        decorr=cfg.data.decorr,
        rng=np.random.default_rng(seed=cfg.seed),
    )
    return dist
    

def setup_plot(cfg: DictConfig) -> Callable:
    """Set up plot function from config."""
    plot_proj = mf.train.plot.PlotProj1D(
        kind="line",
        maxcols=7,
    )
    plot_dist = mf.train.plot.PlotDist2D(
        fig_kws=None,
        bins=cfg.eval.bins,
        limits=(2 * [(-cfg.eval.xmax, +cfg.eval.xmax)])
    )
    plot = mf.train.plot.PlotModel(
        dist=make_dist(cfg), 
        n_samples=cfg.eval.size, 
        plot_proj=plot_proj, 
        plot_dist=plot_dist, 
        device=cfg.device,
    )
    return plot


def setup_eval(cfg: DictConfig) -> Callable:
    """Set up eval function from config."""
    
    def eval(model): 
        return None
        
    return eval
