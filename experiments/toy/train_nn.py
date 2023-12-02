"""Train 2D neural network model with emittance entropy estimator."""
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
        "rings",
        "spirals",
        "swissroll",
        "waterbag",
    ]
)
parser.add_argument("--data-size", type=int, default=int(1.00e+06))
parser.add_argument("--data-noise", type=float, default=None)
parser.add_argument("--data-decorr", type=int, default=0)
parser.add_argument("--data-warp", type=int, default=0)
parser.add_argument("--meas", type=int, default=6)
parser.add_argument("--meas-angle", type=int, default=180.0)
parser.add_argument("--meas-bins", type=int, default=75)
parser.add_argument("--meas-noise", type=float, default=None)
parser.add_argument("--xmax", type=float, default=3.0)

# Neural network generator parameters
parser.add_argument("--hidden-units", type=int, default=128)
parser.add_argument("--hidden-layers", type=int, default=5)
parser.add_argument("--activation", type=str, default="tanh")
parser.add_argument("--dropout", type=float, default=0.0)
parser.add_argument("--base-scale", type=float, default=1.0)
parser.add_argument("--targ-scale", type=float, default=1.0)
parser.add_argument("--entest", type=str, default="cov")

# Optimization
parser.add_argument("--disc", type=str, default="kld", choices=["kld", "mae", "mse"])
parser.add_argument("--mu", type=float, default=5.0)
parser.add_argument("--mu-step", type=float, default=10.0)
parser.add_argument("--mu-scale", type=float, default=1.0)
parser.add_argument("--mu-max", type=float, default=None)
parser.add_argument("--rtol", type=float, default=0.0)
parser.add_argument("--atol", type=float, default=0.0)
parser.add_argument("--cmax", type=float, default=0.0)
parser.add_argument("--absent", type=int, default=0, help="use absolute entropy")

parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--iters", type=int, default=500)
parser.add_argument("--batch-size", type=int, default=40000)

# Optimizer
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--lr-min", type=float, default=0.0005)
parser.add_argument("--lr-drop", type=float, default=0.5)
parser.add_argument("--lr-patience", type=int, default=500)
parser.add_argument("--weight-decay", type=float, default=0.0)

# Diagnostics
parser.add_argument("--check-freq", type=int, default=None)
parser.add_argument("--vis-freq", type=int, default=None)
parser.add_argument("--vis-size", type=int, default=int(1.00e+06))
parser.add_argument("--vis-bins", type=int, default=125)
parser.add_argument("--vis-res", type=int, default=250)
parser.add_argument("--fig-ext", type=str, default="png")
parser.add_argument("--fig-dpi", type=float, default=300)

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

# Generate input distribution.
d = 2
x0 = mf.data.toy.gen_data(
    name=args.data,
    size=args.data_size,
    noise=args.data_noise,
    shuffle=True,
    decorr=args.data_decorr,
    seed=args.seed,
    warp=args.data_warp,
)
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
    entropy_estimator = mf.entropy.CovarianceEntropyEstimator()
elif args.entest == "knn":
    entropy_estimator = mf.entropy.KNNEntropyEstimator(k=5)
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
    diagnostic=diagnostic,
    measurements=measurements,
    penalty_parameter=args.mu,
    discrepancy_function=args.disc,
)
    
# Save config for evaluation.
cfg = {
    "flow": {
        "input_features": d,
        "output_features": d,
        "hidden_units": args.hidden_units,
        "hidden_layers": args.hidden_layers,
        "dropout": args.dropout,
        "activation": args.activation,
    },
    "data": {
        "name": args.data,
        "size": args.data_size,
        "noise": args.data_noise,
        "shuffle": True,
        "decorr": args.data_decorr,
        "seed": args.seed,
        "warp": args.data_warp,
    },
}

with open(man.get_filename("config.pkl"), "wb") as file:
    pickle.dump(cfg, file)


# Diagnostics
# --------------------------------------------------------------------------------------


def make_plots(x, predictions):
    figs = []

    # Plot the ground-truth samples, model samples.
    fig, axs = pplt.subplots(
        ncols=2, 
        space=0.0,
        xspineloc="neither",
        yspineloc="neither",
        share=False
    )
    limits = 2 * [(-xmax, xmax)]
    n_bins = args.vis_bins
    axs[0].hist2d(grab(x0[:args.vis_size])[:, 0], grab(x0[:args.vis_size])[:, 1], bins=n_bins)
    axs[1].hist2d(x[:, 0], x[:, 1], bins=n_bins)
    figs.append(fig)

    # Plot overlayed simulated/measured projections.
    fig, axs = plotting.plot_proj(
        measurements_np, 
        predictions, 
        bin_edges=grab(bin_edges), 
        maxcols=7,
    )
    figs.append(fig)

    return figs
     

def plotter(model):
    x = cvt(model.sample(args.vis_size))
    predictions = model.simulate(x, kde=False)
    predictions = [grab(prediction) for prediction in predictions]
    return make_plots(grab(x), predictions)


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
    penalty_parameter_step=args.penalty,
    penalty_parameter_scale=args.penalty_scale,
    penalty_parameter_max=args.penalty_max,
    save=True,
    vis_freq=args.vis_freq,
    checkpoint_freq=args.check_freq,
    savefig_kws=dict(ext=args.fig_ext, dpi=args.fig_dpi),
)

print(man.timestamp)