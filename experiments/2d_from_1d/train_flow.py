"""Train 2D normalizing flow on linear 1D projections."""
import argparse
import copy
import functools
import inspect
import logging
import os
import pathlib
import pickle
import shutil
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import proplot as pplt
import zuko

import mentflow as mf
from mentflow.utils import grab
from mentflow.utils import unravel
from mentflow.wrappers import WrappedZukoFlow

# Local
import plotting
import utils


# Plot settings
pplt.rc["cmap.discrete"] = False
pplt.rc["cmap.sequential"] = "viridis"
pplt.rc["cycle"] = "538"
pplt.rc["grid"] = False


# Arguments
# --------------------------------------------------------------------------------------
parser = argparse.ArgumentParser()

parser.add_argument("--device", type=str, default="cpu")
parser.add_argument("--seed", type=int, default=None)

# Data
parser.add_argument(
    "--data",
    type=str,
    default="swissroll",
    choices=[
        "circles",
        "gaussians",
        "hollow",
        "kv",
        "pinwheel",
        "spirals",
        "swissroll",
        "waterbag",
    ],
)
parser.add_argument("--data-decorr", type=int, default=0)
parser.add_argument("--data-noise", type=float, default=None)
parser.add_argument("--data-size", type=int, default=int(1.00e06))
parser.add_argument("--data-warp", type=int, default=0)
parser.add_argument("--meas", type=int, default=6)
parser.add_argument("--meas-angle", type=int, default=180.0)
parser.add_argument("--meas-bins", type=int, default=75)
parser.add_argument("--meas-noise", type=float, default=0.0, help="fractional noise")
parser.add_argument("--xmax", type=float, default=3.0)

# Model
parser.add_argument("--transforms", type=int, default=3)
parser.add_argument("--hidden-units", type=int, default=64)
parser.add_argument("--hidden-layers", type=int, default=3)
parser.add_argument("--spline-bins", type=int, default=20)
parser.add_argument("--perm", type=int, default=1)
parser.add_argument("--base-scale", type=float, default=1.0)
parser.add_argument("--prior-scale", type=float, default=1.0)

# Training
parser.add_argument("--batch-size", type=int, default=40000)
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--iters", type=int, default=500)
parser.add_argument("--disc", type=str, default="kld", choices=["kld", "mae", "mse"])
parser.add_argument("--penalty", type=float, default=0.0)
parser.add_argument("--penalty-step", type=float, default=20.0)
parser.add_argument("--penalty-scale", type=float, default=1.1)
parser.add_argument("--penalty-max", type=float, default=None)
parser.add_argument("--rtol", type=float, default=0.0)
parser.add_argument("--atol", type=float, default=0.0)
parser.add_argument("--dmax", type=float, default=0.0)
parser.add_argument("--absent", type=int, default=0, help="use absolute entropy")

# Optimizer (ADAM)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--lr-min", type=float, default=0.001)
parser.add_argument("--lr-drop", type=float, default=0.5)
parser.add_argument("--lr-patience", type=int, default=250)
parser.add_argument("--weight-decay", type=float, default=0.0)

# Diagnostics
parser.add_argument("--check-freq", type=int, default=None)
parser.add_argument("--vis-freq", type=int, default=None)
parser.add_argument("--vis-bins", type=int, default=125)
parser.add_argument("--vis-maxcols", type=int, default=7)
parser.add_argument("--vis-res", type=int, default=250)
parser.add_argument("--vis-size", type=int, default=int(1.00e+06))
parser.add_argument("--vis-line", type=str, default="line", choices=["line", "step"])
parser.add_argument("--fig-dpi", type=float, default=300)
parser.add_argument("--fig-ext", type=str, default="png")

args = parser.parse_args()


# Setup
# --------------------------------------------------------------------------------------

# Create output directories.
path = pathlib.Path(__file__)
filepath = os.path.realpath(__file__)
outdir = os.path.join(path.parent.absolute(), f"output/{args.data}/{path.stem}/")
man = mf.train.ScriptManager(filepath, outdir)
man.make_dirs("checkpoints", "figures")

# Save args and copy of this scripts.
mf.utils.save_pickle(vars(args), man.get_path("args.pkl"))
shutil.copy(__file__, man.get_path("script.py"))

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
    shuffle=True,
    decorr=args.data_decorr,
    rng=rng,
)
mf.utils.save_pickle(dist, man.get_path("dist.pkl"))

# Draw samples from the input distribution.
x0 = dist.sample(args.data_size)
x0 = send(x0)

# Define linear transformations.
angles = np.linspace(0.0, np.radians(args.meas_angle), args.meas, endpoint=False)
transfer_matrices = []
for angle in angles:
    matrix = mf.transform.rotation_matrix(angle)
    matrix = send(matrix)
    transfer_matrices.append(matrix)
transforms = []
for matrix in transfer_matrices:
    transform = mf.transform.Linear(matrix)
    transform = transform.to(device)
    transforms.append(transform)

# Create histogram diagnostic (x axis).
xmax = args.xmax
bin_edges = torch.linspace(-xmax, xmax, args.meas_bins + 1)
bin_edges = send(bin_edges)

diagnostic = mf.diagnostics.Histogram1D(axis=0, bin_edges=bin_edges)
diagnostic = diagnostic.to(device)
diagnostics = [diagnostic]

# Generate training data.
diagnostic.kde = False
measurements = mf.simulate(x0, transforms, diagnostics)
if args.meas_noise:
    for i in range(len(measurements)):
        for j in range(len(measurements[i])):
            measurement = measurements[i][j]
            frac_noise = args.meas_noise * torch.randn(measurement.shape[0])
            frac_noise = send(frac_noise)
            measurement = measurement * (1.0 + frac_noise)
            measurement = torch.clamp(measurement, 0.0, None)
            measurements[i][j] = measurement
diagnostic.kde = True


# Model
# --------------------------------------------------------------------------------------

flow = zuko.flows.NSF(
    features=d,
    transforms=args.transforms,
    bins=args.spline_bins,
    hidden_features=(args.hidden_layers * [args.hidden_units]),
    randperm=args.perm,
)
flow = zuko.flows.Flow(flow.transform.inv, flow.base)  # faster sampling
flow = flow.to(device)
flow = WrappedZukoFlow(flow)

prior = None
if not args.absent:
    prior = zuko.distributions.DiagNormal(
        send(torch.zeros(d)),
        send(args.prior_scale * torch.ones(d)),
    )

entropy_estimator = mf.entropy.MonteCarloEntropyEstimator()

model = mf.MENTFlow(
    generator=flow,
    prior=prior,
    entropy_estimator=entropy_estimator,
    transforms=transforms,
    diagnostics=diagnostics,
    measurements=measurements,
    penalty_parameter=args.penalty,
    discrepancy_function=args.disc,
)
model = model.to(device)

cfg = {
    "generator": {
        "input_features": d,
        "output_features": d,
        "transforms": args.transforms,
        "spline_bins": args.spline_bins,
        "hidden_units": args.hidden_units,
        "hidden_layers": args.hidden_layers,
        "randperm": args.perm,
    },
}
mf.utils.save_pickle(cfg, man.get_path("cfg.pkl"))


# Training diagnostics
# --------------------------------------------------------------------------------------

def make_plots(x, prob, predictions):
    figs = []

    # Plot the true samples, model samples, and model density.
    fig, axs = plotting.plot_dist(
        dist.sample(args.vis_size),
        x,
        prob=prob,
        coords=([np.linspace(-xmax, xmax, s) for s in prob.shape]),
        n_bins=args.vis_bins,
        limits=(2 * [(-xmax, xmax)]),
    )
    figs.append(fig)

    # Plot overlayed simulated/measured projections.    
    fig, axs = plotting.plot_proj(
        [grab(measurement) for measurement in unravel(measurements)],
        predictions,
        bin_edges=grab(diagnostic.bin_edges),
        maxcols=args.vis_maxcols,
        kind=args.vis_line,
    )
    figs.append(fig)

    return figs


def plotter(model):
    # Evaluate the model density on a grid.
    res = args.vis_res
    grid_coords = [np.linspace(-xmax, xmax, res) for i in range(2)]
    grid_points = mf.utils.get_grid_points(grid_coords)
    grid_points = torch.from_numpy(grid_points)
    grid_points = send(grid_points)
    log_prob = model.log_prob(grid_points)
    log_prob = log_prob.reshape((res, res))
    prob = torch.exp(log_prob)

    # Draw samples from the model.
    x = send(model.sample(args.vis_size))

    # Simulate the measurements.
    for diagnostics in model.diagnostics:
        diagnostic.kde = False
        
    predictions = model.simulate(x)
    predictions = [grab(prediction) for prediction in unravel(predictions)]
    
    for diagnostics in model.diagnostics:
        diagnostic.kde = True

    return make_plots(grab(x), grab(prob), predictions)


# FBP/SART benchmarks
# --------------------------------------------------------------------------------------

diagnostic.kde = False

for method in ["sart", "fbp"]:    
    _measurements = [grab(measurement) for measurement in unravel(measurements)]
    prob = utils.reconstruct_tomo(_measurements, angles, method=method, iterations=10)
    edges = 2 * [grab(diagnostic.bin_edges)]
    prob, edges = mf.utils.set_image_shape(prob, edges, (args.vis_res, args.vis_res))
        
    x = mf.utils.sample_hist(prob, edges, n=args.vis_size)
    x = torch.from_numpy(x)
    x = send(x)
    
    predictions = model.simulate(x)
    predictions = [grab(prediction) for prediction in unravel(predictions)]

    figs = make_plots(grab(x), prob, predictions)

    filename = f"fig__test_{method}_00.{args.fig_ext}"
    filename = os.path.join(man.outdir, f"figures/{filename}")
    figs[0].savefig(filename, dpi=args.fig_dpi)

    filename = f"fig__test_{method}_01.{args.fig_ext}"
    filename = os.path.join(man.outdir, f"figures/{filename}")
    figs[1].savefig(filename, dpi=args.fig_dpi)
    plt.close("all")

diagnostic.kde = True


# Training loop
# --------------------------------------------------------------------------------------

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=args.lr,
    weight_decay=args.weight_decay,
)

lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    min_lr=args.lr_min,
    patience=args.lr_patience,
    factor=args.lr_drop,
)

monitor = mf.train.Monitor(model=model, momentum=0.98, freq=1, path=man.get_path("history.pkl"))

trainer = mf.train.Trainer(
    model=model,
    optimizer=optimizer,
    lr_scheduler=lr_scheduler,
    monitor=monitor,
    plotter=plotter,
    output_dir=man.outdir,
    precision=precision,
    device=device,
)

trainer.train(
    epochs=args.epochs,
    iterations=args.iters,
    batch_size=args.batch_size,
    rtol=args.rtol,
    atol=args.atol,
    dmax=args.dmax,
    penalty_step=args.penalty_step,
    penalty_scale=args.penalty_scale,
    penalty_max=args.penalty_max,
    save=True,
    vis_freq=args.vis_freq,
    checkpoint_freq=args.check_freq,
    savefig_kws=dict(ext=args.fig_ext, dpi=args.fig_dpi),
)

print(f"timestamp={man.timestamp}")
