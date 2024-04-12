"""Set up experiments from config."""
import os
import math
import warnings
from typing import Any
from typing import Callable
from typing import List
from typing import Optional
from typing import Type

import torch
import torch.nn as nn
import zuko
from omegaconf import DictConfig
from omegaconf import OmegaConf

import mentflow as mf
from mentflow import unravel


def generate_training_data(
    cfg: DictConfig, 
    make_distribution: Callable,
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
        distribution = make_distribution(cfg)
        x = distribution.sample(cfg.dist.size)
        x = send(x)

        # Simulate measurements.Do not use KDE here.
        for diagnostic in unravel(diagnostics):
            diagnostic.kde = False        
            diagnostic.noise = True

        measurements = mf.simulate.forward(x, transforms, diagnostics)
        
        for diagnostic in unravel(diagnostics):
            diagnostic.kde = True
            diagnostic.noise = False
    
        # Renormalize the measurements in case there was noise.
        for i in range(len(measurements)):
            for j in range(len(measurements[i])):
                measurement = measurements[i][j]
                diagnostic  = diagnostics[i][j]
                volume = 1.0
                if measurement.ndim == 1:
                    volume = diagnostic.edges[1] - diagnostic.edges[0]
                else:
                    volume = math.prod([e[1] - e[0] for e in diagnostic.edges])
                measurement = measurement / torch.sum(measurement) / volume
                measurements[i][j] = measurement

    return (transforms, diagnostics, measurements)


def get_discrepancy_function(name: str) -> Callable:
    discrepancy_function = None
    if name == "kld":
        discrepancy_function = mf.loss.kl_divergence
    elif name == "mae":
        discrepancy_function = mf.loss.mean_absolute_error
    elif name == "mse":
        discrepancy_function = mf.loss.mean_square_error
    else:
        raise ValueError(f"Invalid discrepancy function name {name}")
    return discrepancy_function


def get_entropy_estimator(name: str, prior: Any) -> Callable:
    entropy_estimator = mf.entropy.EmptyEntropyEstimator(prior=prior)
    if name == "cov":
        entropy_estimator = mf.entropy.CovarianceEntropyEstimator(prior=prior)
    elif name == "mc":
        entropy_estimator = mf.entropy.MonteCarloEntropyEstimator(prior=prior)
    return entropy_estimator


def setup_mentflow_model(
    cfg: DictConfig, 
    transforms: List[nn.Module] = None,
    diagnostics: List[List[nn.Module]] = None,
    measurements: List[List[torch.Tensor]] = None,
    device: torch.device = None,
) -> mf.MENTFlow:
    """Setup MENTFlow model from config."""
    if device is None:
        device = cfg.device
        
    def send(x):
        return x.type(torch.float32).to(device)

    # Build generative model.
    kws = OmegaConf.to_container(cfg.gen)
    kws["input_features"]  = cfg.ndim
    kws["output_features"] = cfg.ndim

    # Set default arguments for specific flows.
    if kws["name"] == "nsf":
        kws.setdefault("bins", 20)
    
    generator = mf.generate.build_generator(device=device, **kws)
    generator = generator.to(device)
    
    ### Temp
    if type(generator) is mf.generate.NNGenerator:
        loc = torch.zeros((cfg.ndim,), device=device)
        cov = torch.eye(cfg.ndim, device=device)
        generator.base = torch.distributions.MultivariateNormal(loc, cov)
    
    ## Set Gaussian prior width.
    prior = None
    if cfg.model.prior == "gaussian":
        prior = mf.prior.Gaussian(ndim=cfg.ndim, scale=cfg.model.prior_scale, device=device)

    entropy_estimator = get_entropy_estimator(cfg.model.entropy_estimator, prior)
    
    # Create MENT-Flow model.
    model = mf.MENTFlow(
        generator=generator,
        entropy_estimator=entropy_estimator,
        prior=prior,
        transforms=transforms,
        diagnostics=diagnostics,
        measurements=measurements,
        penalty_parameter=0.0,
        discrepancy_function=get_discrepancy_function(cfg.model.discrepancy)
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


def setup_ment_model(
    cfg: DictConfig, 
    transforms=None,
    diagnostics=None,
    measurements=None,
    device=None,
) -> mf.ment.MENT:
    """Setup MENT model from config."""
    ndim = cfg.ndim
    if device is None:
        device = cfg.device

    # Prior
    prior = None
    if cfg.model.prior == "gaussian":
        prior = mf.prior.Gaussian(
            ndim=ndim, scale=cfg.model.prior_scale, device=device
        )
    if cfg.model.prior == "uniform":
        prior = mf.prior.Uniform(
            ndim=ndim, scale=(10.0 * cfg.model.prior_scale), device=device
        )

    # Sampling    
    kws = OmegaConf.to_container(cfg.model.sampling)
    kws["device"] = device

    grid_xmax = kws.pop("xmax", cfg.eval.xmax)            
    grid_res = kws.pop("res", 45)
    kws["limits"] = ndim * [(-grid_xmax, grid_xmax)]
    kws["shape"] = tuple(ndim * [grid_res])
    
    sampler = mf.sample.GridSampler(**kws)

    # Integration
    integration_limits, integration_shape = None, None
    if measurements is not None:
        integration_limits, integration_shape = [], []
        for index, transform in enumerate(transforms):
            integration_limits.append([])
            integration_shape.append([])
            for diag_index, (diag, meas) in enumerate(zip(diagnostics[index], measurements[index])):
                ndim_int = ndim - meas.ndim    
                
                grid_xmax = cfg.eval.xmax
                if "xmax" in cfg.model.integration:
                    if cfg.model.integration.xmax is not None:
                        grid_xmax = cfg.model.integration.xmax
                    
                limits = ndim_int * [(-grid_xmax, grid_xmax)]
                shape = tuple(ndim_int * [cfg.model.integration.res])
                
                integration_limits[-1].append(limits)
                integration_shape[-1].append(shape)
                
    model = mf.ment.MENT(
        ndim=ndim,
        transforms=transforms,
        diagnostics=diagnostics,
        measurements=measurements,
        discrepancy_function=get_discrepancy_function(cfg.model.discrepancy),
        prior=prior,
        interpolation=cfg.model.interpolation,
        mode=cfg.model.mode,
        integration_limits=integration_limits,
        integration_shape=integration_shape,
        sampler=sampler,
        n_samples=cfg.train.batch_size,
        device=device,
        verbose=cfg.model.verbose,
    )    
    model.to(device)
    return model


def train_ment_model(
    cfg: DictConfig,
    model: mf.ment.MENT,
    setup_plot: Callable,
    setup_eval: Callable,
    output_dir: str,
    notebook: bool = False,
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
        lr=cfg.train.lr,
        dmax=cfg.train.dmax,
        savefig_kws=dict(ext=cfg.plot.ext, dpi=cfg.plot.dpi),
    )





