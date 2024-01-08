"""Train 2D MENT on linear 1D projection.

To do: 
- Write trainer for MENT.
- Update MENT to have same `diagnostics` shape as `MENTFlow`.
"""
import os
import pathlib
from typing import Callable

import hydra
import matplotlib.pyplot as plt
import numpy as np
import ot
import torch
import zuko
from omegaconf import DictConfig

import mentflow as mf
from mentflow.utils import grab
from mentflow.utils import unravel

import setup


@hydra.main(version_base=None, config_path="../../config", config_name="rec_2d_ment.yaml")
def main(cfg: DictConfig):
    print(cfg)

    
    # Output paths
    # --------------------------------------------------------------------------------------
    path = pathlib.Path(__file__)
    output_dir = os.path.join(path.parent.absolute(), f"./output/{cfg.data.name}/")
    man = mf.utils.ScriptManager(os.path.realpath(__file__), output_dir)
    man.save_pickle(cfg, "cfg.pkl")
    man.save_script_copy()

    
    # Data
    # --------------------------------------------------------------------------------------
    transforms, diagnostics, measurements = setup.generate_training_data(
        cfg,
        make_dist=setup.make_dist,
        make_diagnostic=setup.make_diagnostic,
        make_transforms=setup.make_transforms,
    )

    
    # Model
    # --------------------------------------------------------------------------------------
    
    prior = None
    if cfg.model.prior == "gaussian":
        prior = mf.models.ment.GaussianPrior(d=2, scale=cfg.model.prior_scale)
    if cfg.model.prior == "uniform":
        prior = mf.models.ment.UniformPrior(d=2, scale=(10.0 * cfg.model.prior_scale))
    
    sampler_limits = 2 * [(-cfg.meas.xmax, +cfg.meas.xmax)]
    sampler_limits = 1.1 * np.array(sampler_limits)
    sampler = mf.sample.GridSampler(limits=sampler_limits, res=200)
    
    model = mf.models.ment.MENT(
        d=2,
        transforms=transforms,
        measurements=measurements,
        diagnostics=diagnostics[0],  # for now
        discrepancy_function=cfg.model.disc,
        prior=prior,
        sampler=sampler,
        interpolate=cfg.model.interp,  # {"nearest", "linear", "pchip"}
    )
    

    # Training
    # --------------------------------------------------------------------------------------

    # Define simulation method (integration or particle tracking).
    sim_method = cfg.model.method
    sim_kws = {
        "limits": [(-cfg.eval.xmax, +cfg.eval.xmax)],  # integration limits
        "shape": (300,),  # integration grid shape (resolution)
        "n": int(1.00e+06),  # number of particles (if sim_method="sample")
    }
    
    # Make output folders.    
    fig_dir = os.path.join(man.output_dir, f"figures")
    os.makedirs(fig_dir, exist_ok=True)
        
    checkpoint_dir = os.path.join(man.output_dir, f"checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    
    # Training diagnostics:
    dist = setup.make_dist(cfg)


    def plot(model):
        # Generate particles.
        x_true = dist.sample(cfg.eval.size)
        x_pred = model.sample(x_true.shape[0])
    
        # Simulate measurements.
        predictions = mf.sim.forward(x_pred, transforms, diagnostics)
    
        figs = []
        
        ## Plot model vs. true samples.
        fig, axs = mf.train.plot.plot_dist_2d(
            grab(x_true), 
            grab(x_pred), 
            bins=cfg.eval.bins, 
            limits=(2 * [(-cfg.eval.xmax, cfg.eval.xmax)])
        )
        figs.append(fig)
    
        ## Plot measured vs. simulated projections.
        y_meas = [grab(measurement) for measurement in unravel(model.measurements)]
        y_pred = [grab(prediction) for prediction in unravel(predictions)]
        edges = []
        for index, transform in enumerate(transforms):
            for diagnostic in diagnostics[index]:
                edges.append(grab(diagnostic.bin_edges))
                
        fig, axs = mf.train.plot.plot_proj_1d(y_meas, y_pred, edges, kind="line")
        figs.append(fig)
        
        return figs

    
    def eval(model):
        # Draw samples from the model and true distribution.
        x_true = dist.sample(cfg.eval.size)
        x_pred = model.sample(cfg.eval.size)
    
        # Compute simulation-measurement discrepancy
        predictions = mf.sim.forward(x_pred, transforms, diagnostics)        
        discrepancy_vector = model.discrepancy(predictions)
        discrepancy = sum(discrepancy_vector) / len(discrepancy_vector)
    
        # Compute sliced wasserstein distance between true and model samples.
        distance = None
        if cfg.eval.swd:
            n_samples = 50000
            distance = ot.sliced.sliced_wasserstein_distance(
                x_pred[:n_samples],
                x_true[:n_samples], 
                n_projections=50, 
                p=2, 
            )
    
        # Print summary
        print("D(y_model, y_true) = {}".format(discrepancy))
        print("SWD(x_model, x_true) = {}".format(distance))
        return None

    
    # Training loop. (Define an epoch as one Gauss-Seidel iteration.)
    for epoch in range(cfg.train.epochs):
        print(f"epoch = {epoch}")
            
        # Visualize model
        figs = plot(model)
        for index, fig in enumerate(figs):
            if output_dir is not None:
                filename = f"fig_{index:02.0f}_{epoch:03.0f}.{cfg.fig.ext}"
                filename = os.path.join(fig_dir, filename)
                print(f"Saving file {filename}")
                fig.savefig(filename, dpi=cfg.fig.dpi)
            plt.close("all")

        # Evaluate model
        result = eval(model)

        # Save model
        filename = f"checkpoint_{epoch:03.0f}_00000.pt"
        filename = os.path.join(checkpoint_dir, filename)
        print(f"Saving file {filename}")
        model.save(filename)

        # Update lagrange functions.
        model.gauss_seidel_iterate(omega=cfg.train.omega, sim_method=sim_method, **sim_kws)

        print()

    
    print(f"timestamp={man.timestamp}")


if __name__ == "__main__":
    main()
