"""Train 2D MENT-Flow model using the Penalty Method (PM).

For a given lattice, initial distribution, measurement type, and discrepancy function 
(mean absolute value, KL divergence, etc.), a couple setup runs are usually required 
to tune the following parameters (in addition to the flow and optimizer parameters).

The `mu_step` and `mu_scale` parameters determine the penalty parameter step size 
and scaling factor after each epoch. These numbers should be as small as possible to
avoid ill-conditioning. A reasonable choice is to converge in ~10-20 epochs.

The `cmax` parameter defines the convergence condition --- the maximum allowed L1 norm
of the discrepancy vector C, divided by the length of C. Training will cease as soon 
as |C| <= cmax * len(C). The ideal stopping point is usually clear from a plot of |C| 
vs. iteration number. Eventually, large increases in H will be required for very 
small decreases in |C|; we want to stop before this occurs.

Eventually, an automated stopping condition based on the change in C and H may be
implemented.
"""
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
from experiments.toy import plotting
from experiments.toy import utils


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
    ]
)
parser.add_argument("--data-decorr", type=int, default=0)
parser.add_argument("--data-noise", type=float, default=None)
parser.add_argument("--data-size", type=int, default=int(1.00e+06))
parser.add_argument("--data-warp", type=int, default=0)
parser.add_argument("--meas", type=int, default=6)
parser.add_argument("--meas-angle", type=int, default=180.0)
parser.add_argument("--meas-bins", type=int, default=75)
parser.add_argument("--meas-noise", type=float, default=None)
parser.add_argument("--xmax", type=float, default=3.0)

# Model
parser.add_argument("--transforms", type=int, default=3)
parser.add_argument("--hidden-units", type=int, default=64)
parser.add_argument("--hidden-layers", type=int, default=3)
parser.add_argument("--spline-bins", type=int, default=20)
parser.add_argument("--perm", type=int, default=1)
parser.add_argument("--base-scale", type=float, default=1.0)
parser.add_argument("--targ-scale", type=float, default=1.0)

# Training
parser.add_argument("--batch-size", type=int, default=40000)
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--iters", type=int, default=500)
parser.add_argument("--disc", type=str, default="kld", choices=["kld", "mae", "mse"])
parser.add_argument("--penalty", type=float, default=0.0)
parser.add_argument("--penalty-step", type=float, default=5.0)
parser.add_argument("--penalty-scale", type=float, default=1.0)
parser.add_argument("--penalty-max", type=float, default=None)
parser.add_argument("--rtol", type=float, default=0.0)
parser.add_argument("--atol", type=float, default=0.0)
parser.add_argument("--cmax", type=float, default=0.0)
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
parser.add_argument("--fig-dpi", type=float, default=300)
parser.add_argument("--fig-ext", type=str, default="png")

args = parser.parse_args()


# Setup
# --------------------------------------------------------------------------------------

# Set random seed.
rng = np.random.default_rng(seed=args.seed)
if args.seed is not None:
    torch.manual_seed(args.seed)

# Set device and precision.
device = torch.device(args.device)
precision = torch.float32
torch.set_default_dtype(precision)

# Convenience functions
def cvt(x):
    return x.type(precision).to(device)

def grab(x):
    return x.detach().cpu().numpy()


# Create output directories.
path = pathlib.Path(__file__)
filepath = os.path.realpath(__file__)
outdir = os.path.join(
    path.parent.absolute(), 
    f"data_output/{args.data}/{path.stem}/"
)
man = mf.train.ScriptManager(filepath, outdir)
man.make_folders("checkpoints", "figures")

# Create logger.
logger = man.get_logger(filename="log.txt")
logger.info(args)

# Save args.
with open(man.get_filename("args.pkl"), "wb") as file:
    pickle.dump(vars(args), file)

# Save a copy of this script.
shutil.copy(__file__, man.get_filename("script.py"))


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

# Create 1D histogram diagnostic (x axis).
xmax = args.xmax
bin_edges = torch.linspace(-xmax, xmax, args.meas_bins + 1)
bin_edges = cvt(bin_edges)
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

diagnostic = mf.diagnostics.Histogram1D(axis=0, bin_edges=bin_edges)
diagnostic = diagnostic.to(device)

# Perform measurements.
measurements = []
for lattice in lattices:
    measurement = diagnostic(lattice(x0), kde=False)
    if args.meas_noise:
        measurement = measurement + args.meas_noise * torch.randn(measurement.shape[0])
    measurements.append(measurement)
measurements_np = [grab(measurement) for measurement in measurements]


# Model
# --------------------------------------------------------------------------------------

target = None
if not args.absent:
    target = zuko.distributions.DiagNormal(
        cvt(torch.zeros(d)),
        cvt(args.targ_scale * torch.ones(d)),
    )

flow = zuko.flows.NSF(
    features=d, 
    transforms=args.transforms, 
    bins=args.spline_bins,
    hidden_features=(args.hidden_layers * [args.hidden_units]),
    randperm=args.perm,
)
flow = zuko.flows.Flow(flow.transform.inv, flow.base)  # faster sampling
flow = flow.to(device)
    
model = mf.MENTFlow(
    d=d,
    flow=flow,
    target=target,
    lattices=lattices,
    diagnostic=diagnostic,
    measurements=measurements,
    penalty_parameter=args.penalty,
    discrepancy_function=args.disc,
)

# Save config for evaluation.
cfg = {
    "flow": {
        "d": d,
        "transforms": args.transforms,
        "spline_bins": args.spline_bins,
        "hidden_units": args.hidden_units,
        "hidden_layers": args.hidden_layers,
        "randperm": args.perm,
    },
    "dist": dist,
}

with open(man.get_filename("config.pkl"), "wb") as file:
    pickle.dump(cfg, file)


# Diagnostics
# --------------------------------------------------------------------------------------


def make_plots(x, prob, predictions):
    figs = []

    # Plot the ground-truth samples, model samples, and model density.
    fig, axs = plotting.plot_dist(
        grab(x0[:args.vis_size]), 
        x,
        prob=prob, 
        coords=([np.linspace(-xmax, xmax, s) for s in prob.shape]), 
        n_bins=args.vis_bins, 
        limits=(2 * [(-xmax, xmax)]), 
    )
    figs.append(fig)

    # Plot overlayed simulated/measured projections.
    fig, axs = plotting.plot_proj(
        measurements_np,
        predictions, 
        bin_edges=grab(diagnostic.bin_edges), 
        maxcols=args.vis_maxcols,
    )
    figs.append(fig)

    return figs
     

def plotter(model):
    # Evaluate the model density.
    res = args.vis_res
    grid_coords = [np.linspace(-xmax, xmax, res) for i in range(2)]
    grid_points = mf.utils.get_grid_points(grid_coords)
    grid_points = cvt(torch.from_numpy(grid_points))
    log_prob = model.log_prob(grid_points)
    log_prob = log_prob.reshape((res, res))
    prob = torch.exp(log_prob)
    
    # Draw samples from the model.
    x = cvt(model.sample(args.vis_size))

    # Simulate the measurements.
    predictions = model.simulate(x, kde=False)
    predictions = [grab(prediction) for prediction in predictions]

    return make_plots(grab(x), grab(prob), predictions)


# FBP/SART benchmarks
# --------------------------------------------------------------------------------------

for method in ["sart", "fbp"]:
    # Reconstruct image and rescale.
    prob = utils.reconstruct_tomo(measurements_np, angles, method=method, iterations=10)
    coords = 2 * [grab(diagnostic.bin_centers)]
    prob, coords = mf.utils.set_image_shape(prob, coords, (args.vis_res, args.vis_res))

    # Generate samples from the image.
    x = mf.utils.sample_hist(prob, coords=coords, n=args.vis_size)
    x = cvt(torch.from_numpy(x))

    # Simulate the measurements.
    predictions = model.simulate(x, kde=False)
    predictions = [grab(prediction) for prediction in predictions]

    # Make plots.
    figs = make_plots(grab(x), prob, predictions)
    
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

monitor = mf.train.Monitor(
    model=model, 
    momentum=0.98, 
    freq=1,
    path=man.get_filename("history.pkl")
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