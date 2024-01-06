"""Train 2D neural network generator on linear 1D projections."""
import argparse
import copy
import functools
import inspect
import logging
import os
import pathlib
import pickle
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
from mentflow.utils import save_pickle

# Local
import arguments
import plotting
import utils


plotting.set_proplot_rc()


# Arguments
# --------------------------------------------------------------------------------------
parser = arguments.make_parser(model="nn")
args = parser.parse_args()
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
print(model)

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
man.save_pickle(cfg, "cfg.pkl")


# Training
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

trainer = mf.train.Trainer(
    model=model,
    optimizer=optimizer,
    lr_scheduler=lr_scheduler,
    plotter=plotter,
    output_dir=man.output_dir,
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
    vis_freq=args.vis_freq,
    eval_freq=args.eval_freq,
    savefig_kws=dict(ext=args.fig_ext, dpi=args.fig_dpi),
)

print(f"timestamp={man.timestamp}")
