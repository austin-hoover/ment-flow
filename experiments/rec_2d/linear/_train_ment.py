"""Train 2D MENT model."""
import argparse
import os
import pathlib
import pickle
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import ot
import proplot as pplt
import scipy.interpolate
import torch
import zuko
from ipywidgets import interact
from ipywidgets import widgets
from tqdm.notebook import tqdm

import mentflow as mf
from mentflow.utils import grab
from mentflow.utils import unravel

# Local
import arguments
import plotting
import utils


plotting.set_proplot_rc()


# Arguments
# --------------------------------------------------------------------------------------

parser = arguments.make_parser(model="ment")

parser.add_argument("--swd", type=int, default=1)

args = parser.parse_args([])  # not all used
print(args)


# Setup
# --------------------------------------------------------------------------------------

# Create output directories.
path = pathlib.Path(__file__)
output_dir = os.path.join(path.parent.absolute(), f"output/{args.data}/")
man = mf.train.ScriptManager(os.path.realpath(__file__), output_dir)
man.make_dirs("checkpoints", "figures")

# Save args and copy of this script.
man.save_pickle(vars(args), "args.pkl")
man.save_script_copy()

print("output_dir:", man.output_dir)


# Set random seed.
rng = np.random.default_rng(seed=args.seed)
if args.seed is not None:
    torch.manual_seed(args.seed)

# Set device and precision.
device = torch.device(args.device)
precision = torch.float32
torch.set_default_dtype(precision)


def send(x):
    return x.type(precision).to(device)


# Data
# --------------------------------------------------------------------------------------

# Define the input distribution.
d = 2
dist = mf.data.toy.gen_dist(
    args.data,
    noise=args.data_noise,
    decorr=args.data_decorr,
    rng=rng,
)
man.save_pickle(dist, "dist.pkl")

# Draw samples from the input distribution.
x0 = dist.sample(args.data_size)
x0 = send(x0)

# Define transforms.
transforms = utils.make_transforms_rotation(
    args.meas_angle_min, args.meas_angle_max, args.meas_num
)
transforms = [transform.to(device) for transform in transforms]

# Create histogram diagnostic (x axis).
xmax = args.meas_xmax
bin_edges = torch.linspace(-xmax, xmax, args.meas_bins + 1)
bin_edges = send(bin_edges)

diagnostic = mf.diagnostics.Histogram1D(axis=0, bin_edges=bin_edges)
diagnostic = diagnostic.to(device)
diagnostics = [diagnostic]

# Generate training data.
measurements = mf.simulate_nokde(x0, transforms, diagnostics)

# Add measurement noise.
measurements = utils.add_measurement_noise(
    measurements, 
    scale=args.meas_noise, 
    noise_type=args.meas_noise_type, 
    device=device
)


# Model
# --------------------------------------------------------------------------------------

prior = None
if args.prior == "gaussian":
    prior = mf.models.ment.GaussianPrior(d=2, scale=args.prior_scale, device=device)
if args.prior == "uniform":
    prior = mf.models.ment.UniformPrior(d=2, scale=(10.0 * xmax), device=device)

sampler_limits = 2 * [(-xmax, +xmax)]
sampler_limits = 1.1 * np.array(sampler_limits)
sampler = mf.sample.GridSampler(limits=sampler_limits, res=200, device=device)

model = mf.models.ment.MENT(
    d=2,
    transforms=transforms,
    measurements=measurements,
    diagnostics=diagnostics,
    discrepancy_function=args.disc,
    prior=prior,
    sampler=sampler,
    interpolate=args.interpolate,  # {"nearest", "linear", "pchip"}
    device=device,
)


# Training
# --------------------------------------------------------------------------------------

# Make output folders.
output_dir = man.output_dir

if output_dir is not None:
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    fig_dir = os.path.join(output_dir, f"figures")
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
        
    checkpoint_dir = os.path.join(output_dir, f"checkpoints")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)


# Define simulation method (integration or particle tracking).
sim_method = args.method
sim_kws = {
    "limits": [(-xmax, +xmax)],  # integration limits
    "shape": (300,),  # integration grid shape (resolution)
    "n": int(1.00e+06),  # number of particles (if sim_method="sample")
}

# Training diagnostics:

class Plotter:
    def __init__(self):
        return

    def __call__(self, model):
        figs = plotting.plot_model(
            model,
            dist,
            n=args.vis_size, 
            sim_kws=dict(method=sim_method, **sim_kws),
            bins=args.vis_bins,
            xmax=xmax,
            maxcols=args.vis_maxcols, 
            kind=args.vis_line,
            device=device
        )
        return figs


class Evaluator:
    def __init__(self, swd=True):
        self.swd = swd

    def __call__(self, model):
        # Draw samples from the model and true distribution.
        x_true = dist.sample(args.vis_size)
        x_true = send(x_true)
        x = model.sample(args.vis_size)
        x = send(x)
    
        # Compute simulation-measurement discrepancy
        predictions = model.simulate(method=sim_method, **sim_kws)
        discrepancy_vector = model.discrepancy(predictions)
        discrepancy = sum(discrepancy_vector) / len(discrepancy_vector)
    
        # Compute sliced wasserstein distance between true and model samples.
        distance = None
        if self.swd:
            n = 100000
            distance = ot.sliced.sliced_wasserstein_distance(
                x[:n],
                x_true[:n], 
                n_projections=50, 
                p=2, 
            )
    
        # Print summary
        print("D(y_model, y_true) = {}".format(discrepancy))
        print("SWD(x_model, x_true) = {}".format(distance))
        return None


evaluator = Evaluator(swd=args.swd)
plotter = Plotter()


# Training loop. (Define an epoch as one Gauss-Seidel iteration.)

for epoch in range(args.epochs):
    print(f"epoch = {epoch}")

    # Evaluate model
    result = evaluator(model)

    # Save model
    if output_dir is not None:
        filename = f"checkpoint_{epoch:03.0f}_00000.pt"
        filename = os.path.join(checkpoint_dir, filename)
        print(f"Saving file {filename}")
        model.save(filename)

    # Make figures
    figs = plotter(model)
    for index, fig in enumerate(figs):
        if output_dir is not None:
            filename = f"fig_{index:02.0f}_{epoch:03.0f}.{args.fig_ext}"
            filename = os.path.join(fig_dir, filename)
            print(f"Saving file {filename}")
            fig.savefig(filename, dpi=args.fig_dpi)
        plt.close("all")

    # Update Gauss-Seidel
    model.gauss_seidel_iterate(omega=args.omega, sim_method=sim_method, **sim_kws)

print(f"timestamp={man.timestamp}")
