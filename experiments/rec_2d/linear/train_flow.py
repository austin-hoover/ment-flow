import os
import pathlib
import sys

import hydra
import numpy as np
import torch
import zuko
from omegaconf import DictConfig

import mentflow as mf

# local
import plotting
import utils


plotting.set_proplot_rc()


@hydra.main(version_base=None, config_path="../../config", config_name="rec_2d.yaml")
def main(cfg: DictConfig):
    print(cfg)

    
    # Setup
    # --------------------------------------------------------------------------------------
    
    # Create output directories.
    path = pathlib.Path(__file__)
    output_dir = os.path.join(path.parent.absolute(), f"./output/{cfg.data.name}/")
    man = mf.utils.ScriptManager(os.path.realpath(__file__), output_dir)
    man.make_dirs("checkpoints", "figures")
    
    # Save info.
    man.save_pickle(cfg, "cfg.pkl")
    man.save_script_copy()
        
    # Set random seed.
    rng = np.random.default_rng(seed=cfg.seed)
    if cfg.seed is not None:
        torch.manual_seed(cfg.seed)
    
    # Set device and precision.
    device = torch.device(cfg.device)
    precision = torch.float32
    torch.set_default_dtype(precision)

    
    def send(x):
        return x.type(precision).to(device)



    # Data
    # --------------------------------------------------------------------------------------
    
    # Define the input distribution.
    d = 2
    dist = mf.data.toy.gen_dist(
        name=cfg.data.name,
        noise=cfg.data.noise,
        decorr=cfg.data.decorr,
        rng=rng,
    )
    man.save_pickle(dist, "dist.pkl")
    
    # Draw samples from the input distribution.
    x0 = dist.sample(cfg.data.size)
    x0 = send(x0)
    
    # Define transforms.
    transforms = utils.make_rotation_transforms(
        cfg.meas.min_angle, cfg.meas.max_angle, cfg.meas.num,
    )
    transforms = [transform.to(device) for transform in transforms]
    
    # Create histogram diagnostic (x axis).
    xmax = cfg.meas.xmax
    bin_edges = torch.linspace(-xmax, xmax, cfg.meas.bins + 1)
    bin_edges = send(bin_edges)
    
    diagnostic = mf.diagnostics.Histogram1D(axis=0, bin_edges=bin_edges)
    diagnostic = diagnostic.to(device)
    diagnostics = [diagnostic]
    
    # Generate training data.
    measurements = mf.simulate_nokde(x0, transforms, diagnostics)
    
    # Add measurement noise.
    measurements = utils.add_measurement_noise(
        measurements, 
        scale=cfg.meas.noise, 
        noise_type=cfg.meas.noise_type, 
        device=device
    )


    # Model
    # --------------------------------------------------------------------------------------
    
    gen = mf.gen.build_gen(**cfg["gen"])
    gen = gen.to(device)
    
    prior = zuko.distributions.DiagNormal(
        send(torch.zeros(d)),
        send(cfg.model.prior_scale * torch.ones(d)),
    )

    entropy_estimator = mf.entropy.MonteCarloEntropyEstimator()
        
    model = mf.MENTFlow(
        gen=gen,
        entropy_estimator=entropy_estimator,
        prior=prior,
        transforms=transforms,
        diagnostics=diagnostics,
        measurements=measurements,
        penalty_parameter=cfg.train.penalty,
        discrepancy_function=cfg.model.disc,
    )
    model = model.to(device)
    print(model)


    
    # Training
    # --------------------------------------------------------------------------------------    
    
    def evaluator(model):
        return
        

    def plotter(model):
        figs = plotting.plot_model(
            model,
            dist,
            n=cfg.train.vis_size,
            bins=cfg.train.vis_bins,
            xmax=xmax,
            maxcols=cfg.train.vis_maxcols, 
            kind=cfg.train.vis_line,
            device=device
        )
        return figs
    
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.train.lr,
        weight_decay=0.0,  # would need to reset this after each epoch...
    )
    
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        min_lr=cfg.train.lr_min,
        patience=cfg.train.lr_patience,
        factor=cfg.train.lr_drop,
    )
    
    trainer = mf.train.Trainer(
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        plotter=plotter,
        evaluator=evaluator,
        output_dir=man.output_dir,
        precision=precision,
        device=device,
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
        eval_freq=cfg.train.eval_freq,
        savefig_kws=dict(ext=cfg.train.fig_ext, dpi=cfg.train.fig_dpi),
    )
    
    print(f"timestamp={man.timestamp}")



if __name__ == '__main__':
    main()