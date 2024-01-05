"""Train 2D neural network generator on linear 1D projections."""
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

# Local
import plotting


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
    default="two-spirals",
    choices=[
        "eight-gaussians",
        "galaxy",
        "gaussian",
        "hollow",
        "kv",
        "leaf",
        "pinwheel",
        "rings",
        "swissroll",
        "two-spirals",
        "waterbag",
    ],
)
parser.add_argument("--data-decorr", type=int, default=0)
parser.add_argument("--data-noise", type=float, default=None)
parser.add_argument("--data-size", type=int, default=int(1.00e+06))
parser.add_argument("--data-warp", type=int, default=0)
parser.add_argument("--n-meas", type=int, default=6)
parser.add_argument("--meas-angle-min", type=int, default=0.0)
parser.add_argument("--meas-angle-max", type=int, default=180.0)
parser.add_argument("--meas-bins", type=int, default=75)
parser.add_argument("--meas-noise", type=float, default=0.0)
parser.add_argument("--meas-noise-type", type=str, default="gaussian")
parser.add_argument("--meas-xmax", type=float, default=3.0)

# Model
parser.add_argument("--input-features", type=int, default=2)
parser.add_argument("--hidden-units", type=int, default=20)
parser.add_argument("--hidden-layers", type=int, default=2)
parser.add_argument("--activation", type=str, default="tanh")
parser.add_argument("--dropout", type=float, default=0.0)
parser.add_argument("--targ-scale", type=float, default=1.0)
parser.add_argument("--entest", type=str, default="cov")

# Training
parser.add_argument("--batch-size", type=int, default=30000)
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--iters", type=int, default=300)

parser.add_argument("--disc", type=str, default="kld", choices=["kld", "mae", "mse"])
parser.add_argument("--penalty", type=float, default=1000.0)
parser.add_argument("--penalty-step", type=float, default=0.0)
parser.add_argument("--penalty-scale", type=float, default=1.0)
parser.add_argument("--penalty-max", type=float, default=None)

parser.add_argument("--rtol", type=float, default=0.0)
parser.add_argument("--atol", type=float, default=0.0)
parser.add_argument("--dmax", type=float, default=7.50e-04)

# Optimizer (ADAM)
parser.add_argument("--lr", type=float, default=0.01)
parser.add_argument("--lr-min", type=float, default=0.001)
parser.add_argument("--lr-drop", type=float, default=0.1)
parser.add_argument("--lr-patience", type=int, default=400)

# Training diagnostics
parser.add_argument("--check-freq", type=int, default=None)
parser.add_argument("--vis-freq", type=int, default=None)
parser.add_argument("--vis-bins", type=int, default=125)
parser.add_argument("--vis-maxcols", type=int, default=7)
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
    decorr=args.data_decorr,
    rng=rng,
)
mf.utils.save_pickle(dist, man.get_path("dist.pkl"))

# Draw samples from the input distribution.
x0 = dist.sample(args.data_size)
x0 = send(x0)

# Define transforms.
angles = np.linspace(
    np.radians(args.meas_angle_min), 
    np.radians(args.meas_angle_max), 
    args.n_meas, 
    endpoint=False
)
transforms = []
for angle in angles:
    matrix = mf.transform.rotation_matrix(angle)
    matrix = send(matrix)
    transform = mf.transform.LinearTransform(matrix)
    transform = transform.to(device)
    transforms.append(transform)

# Create histogram diagnostic (x axis).
xmax = args.meas_xmax
bin_edges = torch.linspace(-xmax, xmax, args.meas_bins + 1)
bin_edges = send(bin_edges)

diagnostic = mf.diagnostics.Histogram1D(axis=0, bin_edges=bin_edges)
diagnostic = diagnostic.to(device)
diagnostics = [diagnostic]

# Generate training data.
diagnostic.kde = False
measurements = mf.simulate(x0, transforms, diagnostics)
diagnostic.kde = True

# Add noise to measurements.
if args.meas_noise:
    for i in range(len(measurements)):
        for j in range(len(measurements[i])):
            measurement = measurements[i][j]

            frac_noise = torch.zeros(measurement.shape[0])
            if meas_noise_type == "uniform":
                frac_noise = meas_noise * torch.rand(measurement.shape[0]) * 2.0
            else:
                frac_noise = meas_noise * torch.randn(measurement.shape[0])
            frac_noise = send(frac_noise)
            measurement = measurement * (1.0 + frac_noise)
            measurement = torch.clamp(measurement, 0.0, None)
            measurements[i][j] = measurement


# Model
# --------------------------------------------------------------------------------------

transformer = mf.models.NNTransformer(
    input_features=args.input_features,
    output_features=d,
    hidden_layers=args.hidden_layers,
    hidden_units=args.hidden_units,
    dropout=args.dropout,
    activation=args.activation,
)
base = torch.distributions.Normal(
    send(torch.zeros(args.input_features)),
    send(torch.ones(args.input_features)),
)
generator = mf.models.NNGenerator(base, transformer)
generator = generator.to(device)

entropy_estimator = mf.entropy.EmptyEntropyEstimator()
if args.entest == "cov":
    entropy_estimator = mf.entropy.CovarianceEntropyEstimator()
if args.entest == "knn":
    entropy_estimator = mf.entropy.KNNEntropyEstimator(k=5)

model = mf.MENTFlow(
    generator=generator,
    prior=None,
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
        "hidden_units": args.hidden_units,
        "hidden_layers": args.hidden_layers,
        "dropout": args.dropout,
        "activation": args.activation,
    },
}
mf.utils.save_pickle(cfg, man.get_path("cfg.pkl"))

print(model)


# Training diagnostics
# --------------------------------------------------------------------------------------

def plotter(model):

    figs = plotting.plot_model(
        model,
        dist,
        n=args.vis_size, 
        bins=args.vis_bins,
        xmax=xmax,
        maxcols=args.vis_maxcols, 
        kind=args.vis_line,
        device=device
    )
    return figs


# Training loop
# --------------------------------------------------------------------------------------

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=args.lr,
    weight_decay=0.0,  # would need to reset this after each epoch...
)

lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    min_lr=args.lr_min,
    patience=args.lr_patience,
    factor=args.lr_drop,
)

monitor = mf.train.Monitor(
    model=model, 
    momentum=0.9,
    freq=1, 
    path=man.get_path("history.pkl")
)

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