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
from mentflow.utils import unravel

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
parser.add_argument("--meas-noise", type=float, default=None)
parser.add_argument("--xmax", type=float, default=3.0)

# Model
parser.add_argument("--hidden-units", type=int, default=128)
parser.add_argument("--hidden-layers", type=int, default=5)
parser.add_argument("--activation", type=str, default="tanh")
parser.add_argument("--dropout", type=float, default=0.0)
parser.add_argument("--base-scale", type=float, default=1.0)
parser.add_argument("--targ-scale", type=float, default=1.0)
parser.add_argument("--entest", type=str, default="cov")

# Training
parser.add_argument("--batch-size", type=int, default=40000)
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--iters", type=int, default=500)
parser.add_argument("--disc", type=str, default="kld", choices=["kld", "mae", "mse"])
parser.add_argument("--penalty", type=float, default=1000.0)
parser.add_argument("--penalty-step", type=float, default=0.0)
parser.add_argument("--penalty-scale", type=float, default=1.0)
parser.add_argument("--penalty-max", type=float, default=None)
parser.add_argument("--rtol", type=float, default=0.0)
parser.add_argument("--atol", type=float, default=0.0)
parser.add_argument("--cmax", type=float, default=0.0)
parser.add_argument("--absent", type=int, default=1, help="use absolute entropy")

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
parser.add_argument("--vis-size", type=int, default=int(1.00e06))
parser.add_argument("--fig-dpi", type=float, default=300)
parser.add_argument("--fig-ext", type=str, default="png")

args = parser.parse_args()


# Setup
# --------------------------------------------------------------------------------------

# Create output directories.
path = pathlib.Path(__file__)
filepath = os.path.realpath(__file__)
outdir = os.path.join(path.parent.absolute(), f"data_output/{args.data}/{path.stem}/")
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


def cvt(x):
    return x.type(precision).to(device)


def grab(x):
    return x.detach().cpu().numpy()


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
x0 = cvt(torch.from_numpy(x0))

# Generate lattices.
angles = np.linspace(0.0, np.radians(args.meas_angle), args.meas, endpoint=False)
transfer_matrices = []
for angle in angles:
    matrix = mf.utils.rotation_matrix(angle)
    matrix = cvt(torch.from_numpy(matrix))
    transfer_matrices.append(matrix)
lattices = []
for matrix in transfer_matrices:
    lattice = mf.lattice.LinearLattice()
    lattice = lattice.to(device)
    lattice.set_matrix(matrix)
    lattices.append(lattice)

# Create histogram diagnostic (x axis).
xmax = args.xmax
bin_edges = torch.linspace(-xmax, xmax, args.meas_bins + 1)
bin_edges = cvt(bin_edges)
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

diagnostic = mf.diagnostics.Histogram1D(axis=0, bin_edges=bin_edges)
diagnostic = diagnostic.to(device)
diagnostics = [diagnostic]

# Perform measurements.
measurements = []
for lattice in lattices:
    measurements.append([])
    for diagnostic in diagnostics:
        measurement = diagnostic(lattice(x0), kde=False)
        if args.meas_noise:
            measurement = measurement + args.meas_noise * torch.randn(measurement.shape[0])
        measurements[-1].append(measurement)


# Model
# --------------------------------------------------------------------------------------

flow = mf.models.NNGenerator(
    input_features=d,
    output_features=d,
    hidden_layers=args.hidden_layers,
    hidden_units=args.hidden_units,
    dropout=args.dropout,
    activation=args.activation,
)
flow = flow.to(device)

target = None
if not args.absent:
    target = torch.distributions.Normal(
        loc=cvt(torch.zeros(d)),
        scale=cvt(args.base_scale * torch.ones(d)),
    )

base = torch.distributions.Normal(
    loc=cvt(torch.zeros(d)),
    scale=cvt(args.base_scale * torch.ones(d)),
)

if args.entest == "cov":
    entropy_estimator = mf.entropy.CovarianceEntropyEstimator(prior=target)
elif args.entest == "knn":
    entropy_estimator = mf.entropy.KNNEntropyEstimator(k=5, prior=target)
else:
    entropy_estimator = mf.entropy.EmptyEntropyEstimator()
entropy_estimator = entropy_estimator.to(device)

model = mf.MENTNN(
    d=d,
    flow=flow,
    base=base,
    target=target,
    entropy_estimator=entropy_estimator,
    lattices=lattices,
    diagnostics=diagnostics,
    measurements=measurements,
    penalty_parameter=args.penalty,
    discrepancy_function=args.disc,
)

cfg = {
    "flow": {
        "input_features": d,
        "output_features": d,
        "hidden_units": args.hidden_units,
        "hidden_layers": args.hidden_layers,
        "dropout": args.dropout,
        "activation": args.activation,
    },
    "base": base,
    "entropy_estimator": entropy_estimator,
}
mf.utils.save_pickle(cfg, man.get_path("cfg.pkl"))


def make_plots(x, predictions):
    figs = []

    # Plot the true samples and model samples.
    fig, axs = plotting.plot_dist(
        dist.sample(args.vis_size),
        x,
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
    )
    figs.append(fig)

    return figs


def plotter(model):
    x = cvt(model.sample(args.vis_size))
    predictions = model.simulate(x, kde=False)
    predictions = [grab(prediction) for prediction in unravel(predictions)]
    return make_plots(grab(x), predictions)


# FBP/SART benchmarks
# --------------------------------------------------------------------------------------

for method in ["sart", "fbp"]:
    _measurements = [grab(measurement) for measurement in unravel(measurements)]
    prob = utils.reconstruct_tomo(_measurements, angles, method=method, iterations=10)
    coords = 2 * [grab(diagnostic.bin_centers)]
    prob, coords = mf.utils.set_image_shape(prob, coords, (args.vis_res, args.vis_res))

    x = mf.utils.sample_hist(prob, coords=coords, n=args.vis_size)
    x = cvt(torch.from_numpy(x))

    predictions = model.simulate(x, kde=False)
    _predictions = [grab(prediction) for prediction in unravel(predictions)]

    figs = make_plots(grab(x), _predictions)

    filename = f"fig__test_{method}_00.{args.fig_ext}"
    filename = os.path.join(man.outdir, f"figures/{filename}")
    figs[0].savefig(filename, dpi=args.fig_dpi)

    filename = f"fig__test_{method}_01.{args.fig_ext}"
    filename = os.path.join(man.outdir, f"figures/{filename}")
    figs[1].savefig(filename, dpi=args.fig_dpi)
    plt.close("all")


# Training
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
    cmax=args.cmax,
    penalty_step=args.penalty_step,
    penalty_scale=args.penalty_scale,
    penalty_max=args.penalty_max,
    save=True,
    vis_freq=args.vis_freq,
    checkpoint_freq=args.check_freq,
    savefig_kws=dict(ext=args.fig_ext, dpi=args.fig_dpi),
)

print(man.timestamp)
