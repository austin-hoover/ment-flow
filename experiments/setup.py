"""Tools to set up experiments from config."""
import os
import math
import warnings
from typing import Callable
from typing import List
from typing import Optional
from typing import Type

import torch
import zuko
from omegaconf import DictConfig
from omegaconf import OmegaConf

import mentflow as mf
from mentflow import unravel


def setup_mentflow_model(
    cfg: DictConfig, 
    transforms=None,
    diagnostics=None,
    measurements=None,
    device=None,
):
    """Setup MENTFlow model from config."""
    if device is None:
        device = cfg.device
        
    def send(x):
        return x.type(torch.float32).to(device)

    # Build generative model.
    kws = OmegaConf.to_container(cfg.gen)
    kws["input_features"]  = cfg.d
    kws["output_features"] = cfg.d

    # Set default arguments for specific flows.
    if kws["name"] == "nsf":
        kws.setdefault("bins", 20)
    
    gen = mf.gen.build_gen(device=device, **kws)
    gen = gen.to(device)
    
    
    ## Temp
    if type(gen) is mf.gen.NNGen:
        loc = torch.zeros((cfg.d,), device=device)
        cov = torch.eye(cfg.d, device=device)
        gen.base = torch.distributions.MultivariateNormal(loc, cov)
    

    # Set Gaussian prior width.
    d = cfg.d
    prior = zuko.distributions.DiagNormal(
        send(torch.zeros(d)),
        send(cfg.model.prior_scale * torch.ones(d)),
    )
    
    # Create MENT-Flow model.
    model = mf.MENTFlow(
        gen=gen,
        entropy_estimator=cfg.model.entest,
        prior=prior,
        transforms=transforms,
        diagnostics=diagnostics,
        measurements=measurements,
        penalty_parameter=0.0,
        discrepancy_function=cfg.model.disc,
    )
    model = model.to(device)
    return model


def train_mentflow_model(
    cfg: DictConfig,
    model: Type[torch.nn.Module],
    setup_plot: Optional[Callable] = None,
    setup_eval: Optional[Callable] = None,
    output_dir=None,
    notebook=False,
) -> None:
    """Set up MENT-Flow trainer from config, then train the model."""
    if cfg.seed is not None:
        torch.manual_seed(cfg.seed)
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.train.lr,
        weight_decay=0.0,
    )
    
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        min_lr=cfg.train.lr_min,
        patience=cfg.train.lr_patience,
        factor=cfg.train.lr_drop,
    )

    plot = None
    if setup_plot is not None:
        plot = setup_plot(cfg)

    _eval = None
    if setup_eval is not None:
        _eval = setup_eval(cfg)
   
    trainer = mf.train.Trainer(
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        plot=plot,
        eval=_eval,
        output_dir=output_dir,
        notebook=notebook,
        load_best=cfg.train.load_best,
    )
    
    trainer.train(
        epochs=cfg.train.epochs,
        iterations=cfg.train.iters,
        batch_size=cfg.train.batch_size,
        rtol=cfg.train.rtol,
        atol=cfg.train.atol,
        dmax=cfg.train.dmax,
        penalty_start=cfg.train.penalty,
        penalty_step=cfg.train.penalty_step,
        penalty_scale=cfg.train.penalty_scale,
        penalty_max=cfg.train.penalty_max,
        eval_freq=cfg.eval.freq,
        savefig_kws=dict(ext=cfg.plot.ext, dpi=cfg.plot.dpi),
    )


def generate_training_data(
    cfg: DictConfig, 
    make_dist: Callable,
    make_diagnostics: Callable,
    make_transforms: Callable, 
):
    """Generate training data (tranforms, diagnostics, measurements) from config.

    This function creates the same set of diagnostic for each transform.
    """
    with torch.no_grad():
        device = torch.device(cfg.device)
        send = lambda x: x.type(torch.float32).to(device)
    
        if cfg.seed is not None:
            torch.manual_seed(cfg.seed)
        
        # Define transforms.
        transforms = make_transforms(cfg)
        transforms = [transform.to(device) for transform in transforms]
        
        # Create histogram diagnostic.
        diagnostics = make_diagnostics(cfg)
        diagnostics = [diagnostics for transform in transforms]
    
        # Generate samples from input distribution.
        dist = make_dist(cfg)
        x = dist.sample(cfg.dist.size)
        x = send(x)

        # Simulate measurements.
        for diagnostic in unravel(diagnostics):
            diagnostic.kde = False        
            diagnostic.noise = True


        measurements = mf.sim.forward(x, transforms, diagnostics)
        
        for diagnostic in unravel(diagnostics):
            diagnostic.kde = True
            diagnostic.noise = False
    
        # Renormalize measurements in case there was noise.
        for i in range(len(measurements)):
            for j in range(len(measurements[i])):
                measurement = measurements[i][j]
                diagnostic  = diagnostics[i][j]
                bin_volume = 1.0
                if measurement.ndim == 1:
                    bin_volume = diagnostic.bin_edges[1] - diagnostic.bin_edges[0]
                else:
                    bin_volume = math.prod([e[1] - e[0] for e in diagnostic.bin_edges])
                measurement = measurement / measurement.sum() / bin_volume
                measurements[i][j] = measurement

    return (transforms, diagnostics, measurements)


def setup_ment_model(
    cfg: DictConfig, 
    transforms=None,
    diagnostics=None,
    measurements=None,
    device=None,
):
    """Setup MENT model from config."""
    d = cfg.d
    if device is None:
        device = cfg.device

    # Prior
    prior = None
    if cfg.model.prior == "gaussian":
        prior = mf.alg.ment.GaussianPrior(d=d, scale=cfg.model.prior_scale, device=device)
    if cfg.model.prior == "uniform":
        prior = mf.alg.ment.UniformPrior(d=d, scale=(10.0 * cfg.model.prior_scale), device=device)

    # Sampling
    sampler = None
    kws = OmegaConf.to_container(cfg.model.samp)
    kws["device"] = device
    method = kws.pop("method", "grid")
    if method in ["grid", "slicegrid"]:
        grid_xmax = kws.pop("xmax", cfg.eval.xmax)            
        grid_res = kws.pop("res", 45)
        kws["grid_limits"] = d * [(-grid_xmax, grid_xmax)]
        kws["grid_shape"] = tuple(d * [grid_res])
        if method == "grid":
            sampler = mf.sample.GridSampler(**kws)
        elif method == "slicegrid":
            sampler = mf.sample.SliceGridSampler(**kws)

    # Integration
    integration_grid_limits = None
    integration_grid_shape = None
    if measurements is not None:
        integration_grid_limits = []
        integration_grid_shape = []
        for index, transform in enumerate(transforms):
            integration_grid_limits.append([])
            integration_grid_shape.append([])
            for diag_index, (diagnostic, measurement) in enumerate(zip(diagnostics[index], measurements[index])):
                d_int = d - measurement.ndim
                grid_xmax = cfg.eval.xmax
                if "xmax" in cfg.model.int:
                    grid_xmax = cfg.model.int["xmax"]
                grid_limits = d_int * [(-grid_xmax, grid_xmax)]
                grid_shape = tuple(d_int * [cfg.model.int.res])
                integration_grid_limits[-1].append(grid_limits)
                integration_grid_shape[-1].append(grid_shape)
                
    model = mf.alg.ment.MENT(
        d=d,
        transforms=transforms,
        measurements=measurements,
        diagnostics=diagnostics,
        discrepancy_function=cfg.model.disc,
        prior=prior,
        sampler=sampler,
        interpolate=cfg.model.interp,
        mode=cfg.model.mode,
        integration_grid_limits=integration_grid_limits,
        integration_grid_shape=integration_grid_shape,
        n_samples=cfg.train.batch_size,
        device=device,
        verbose=cfg.model.verbose,
    )
    model.to(device)
    return model


def train_ment_model(
    cfg=None,
    model=None,
    setup_plot=None,
    setup_eval=None,
    output_dir=None,
    notebook=False,
) -> None:
    """Set up MENT trainer from config, then train the model."""
    if cfg.seed is not None:
        torch.manual_seed(cfg.seed)
    
    plot = None
    if setup_plot is not None:
        plot = setup_plot(cfg)
        
    _eval = None
    if setup_eval is not None:
        _eval = setup_eval(cfg)
   
    trainer = mf.train.MENTTrainer(
        model=model,
        eval=_eval,
        plot=plot,
        output_dir=output_dir,
        notebook=notebook,
    )
    trainer.train(
        epochs=cfg.train.epochs, 
        omega=cfg.train.omega,
        dmax=cfg.train.dmax,
        savefig_kws=dict(ext=cfg.plot.ext, dpi=cfg.plot.dpi),
    )
