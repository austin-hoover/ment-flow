"""Tools to set up experiments from config."""
from typing import Callable
from typing import List
from typing import Optional
from typing import Type

import torch
import zuko
from omegaconf import DictConfig

import mentflow as mf
from mentflow import unravel


def setup_model(
    cfg: DictConfig, 
    transforms=None,
    diagnostics=None,
    measurements=None,
):
    """Setup MENTFlow model from config."""
    
    device = torch.device(cfg.device)
    send = lambda x: x.type(torch.float32).to(device)

    # Build generative model.
    gen = mf.gen.build_gen(device=device, **cfg.gen)
    gen = gen.to(device)

    # Set Gaussian prior width.
    d = cfg.gen.output_features
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
    
    
def setup_and_run_trainer(
    cfg: DictConfig,
    model: Type[torch.nn.Module],
    setup_plot: Optional[Callable] = None,
    setup_eval: Optional[Callable] = None,
    output_dir=None,
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

    eval = None
    if setup_eval is not None:
        eval = setup_eval(cfg)
   
    trainer = mf.train.Trainer(
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        plot=plot,
        eval=eval,
        output_dir=output_dir,
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
        savefig_kws=dict(ext=cfg.fig.ext, dpi=cfg.fig.dpi),
    )
    

def generate_training_data(
    cfg: DictConfig, 
    make_dist: Callable,
    make_diagnostic: Callable,
    make_transforms: Callable, 
):
    """Generate training data (tranforms, diagnostics, measurements) from config.

    This function creates the same diagnostic for each transform.
    """
    device = torch.device(cfg.device)
    send = lambda x: x.type(torch.float32).to(device)
    
    # Define transforms.
    transforms = make_transforms(cfg)
    transforms = [transform.to(device) for transform in transforms]
    
    # Create histogram diagnostic.
    diagnostic = make_diagnostic(cfg)
    diagnostics = [[diagnostic,] for transform in transforms]

    # Generate samples from input distribution.
    dist = make_dist(cfg)
    x = dist.sample(cfg.data.size)
    x = send(x)

    # Simulate measurements.
    for diagnostic in unravel(diagnostics):
        diagnostic.kde = False        
        
    measurements = mf.sim.forward(x, transforms, diagnostics)
    
    for diagnostic in unravel(diagnostics):
        diagnostic.kde = True

    return (transforms, diagnostics, measurements)