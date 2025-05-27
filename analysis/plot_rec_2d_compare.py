"""Compare 2D reconstruction vs. number of projections (MENT, MENT-Flow, NN)."""
import os
import pathlib
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import psdist.plot as psv
import seaborn as sns
import torch
import ultraplot as uplt
from tqdm import tqdm
from tqdm import trange

import mentflow as mf
from mentflow.utils import grab
from mentflow.utils import unravel
from mentflow.utils import load_pickle

from utils import cubehelix_cmap

sys.path.append("..")
from experiments.setup import setup_mentflow_model
from experiments.setup import setup_ment_model
from experiments.rec_2d.setup import make_distribution
from experiments.load import epoch_and_iteration_number
from experiments.load import list_contents


uplt.rc["cmap.discrete"] = False
uplt.rc["cmap.sequential"] = "viridis"
uplt.rc["figure.facecolor"] = "white"
uplt.rc["grid"] = False


# Settings
# ----------------------------------------------------------------------------------------

path = pathlib.Path(__file__)

input_dir = "../experiments/rec_2d/outputs/"
output_dir = "./outputs"
os.makedirs(output_dir, exist_ok=True)

device = torch.device("cpu")
precision = torch.float32
send = lambda x: x.type(precision).to(device)


# Load data
# ----------------------------------------------------------------------------------------

# Load all run info. Keep the results for the selected distribution.
dist_names = set()
for key in ["flow", "ment", "nn"]:
    for folder in list_contents(os.path.join(input_dir, f"train_{key}")):
        cfg = load_pickle(os.path.join(folder, "config/config.pickle"))
        dist_names.add(cfg.dist.name)
dist_names = list(dist_names)

print("dist_names:")
print(dist_names)

runs = {}
for dist_name in dist_names:
    runs[dist_name] = {}
    for key in ["flow", "nn", "ment"]:
        runs[dist_name][key] = []
        data_dir = os.path.join(input_dir, f"train_{key}")

        for folder in list_contents(data_dir):
            cfg = load_pickle(os.path.join(folder, "config/config.pickle"))
            if cfg.dist.name != dist_name:
                continue

            history = load_pickle(os.path.join(folder, "history.pkl"))

            checkpoints = []
            for path in list_contents(os.path.join(folder, "checkpoints")):
                epoch, iteration = epoch_and_iteration_number(path)
                checkpoint = {
                    "epoch": epoch,
                    "iteration": iteration,
                    "path": path,
                }
                checkpoints.append(checkpoint)

                print(f"loaded {path}")

            run = {
                "cfg": cfg, 
                "history": history, 
                "checkpoints": checkpoints,
            }
            runs[dist_name][key].append(run)


# Set up the model architecture from the first config. (All runs used the same architecture.)
models = {}
for key in ["flow", "nn", "ment"]:  
    dist_name = dist_names[0]
    index = 0
    cfg = runs[dist_name][key][index]["cfg"]
    if key == "ment":       
        model = setup_ment_model(cfg, transforms=[], diagnostics=[], measurements=[], device=device)
    else:
        model = setup_mentflow_model(cfg, transforms=[], diagnostics=[], measurements=[], device=device)
    model = model.to(device)
    models[key] = model

    print(f"setup {key} architecture")


# Compare all models and runs
# ----------------------------------------------------------------------------------------

# Get the number of runs per model.
n_runs = len(runs[dist_names[0]]["flow"])

# Get the max number of measurements.
run_index = -1
run = runs[dist_names[0]]["flow"][run_index]
cfg = run["cfg"]
max_n_meas = cfg.meas.num

print("n_runs =", n_runs)
print("max_n_meas =", max_n_meas)


def plot_compare_all_profiles(
    dist_name: str, 
    n_samples: int, 
    xmax: float, 
    bins: int, 
    fig_kws: dict = None,
    plot_kws: dict = None,
):
    n_models = 4
    n_rows = n_models + max_n_meas
    n_cols = n_runs
    height_ratio = 0.5
    ymax = 1.425

    fig, axs = uplt.subplots(
        nrows=n_rows,
        ncols=n_cols,
        share=False,
        height_ratios=(n_models * [1.0] + max_n_meas * [height_ratio]),
        space=1.0,
        figwidth=10.0,
    )
    axs.format(
        xticks=[],
        yticks=[],
        xspineloc="neither",
        yspineloc="neither",
    )
    axs[n_models:, :].format(xspineloc="bottom")
    
    cmaps = [
        uplt.Colormap("mono", right=0.95),
        uplt.Colormap("mono", right=0.95),
        uplt.Colormap("mono", right=0.95),
        uplt.Colormap("mono", right=0.95),
    ]

    lineplot_kws = [
        dict(color="black", ls=":", lw=0.9, zorder=0),
        dict(color="black", ls="-", lw=0.9, zorder=0),
        dict(color="black", ls="-", lw=0.9, zorder=0),
        dict(color="black", ls="-", lw=0.9, zorder=0),
    ]

    if plot_kws is None:
        plot_kws = dict()

    with torch.no_grad():
        for run_index in range(n_runs):
            # Load model parameters.
            for key in runs[dist_name]:
                run = runs[dist_name][key][run_index]
                models[key].load(run["checkpoints"][-1]["path"], device=device)

            # Distribution is the same for all models.
            cfg = runs[dist_name]["flow"][run_index]["cfg"]
            dist = make_distribution(cfg)

            # Plot samples from model.
            _models = [dist] + [models[key] for key in ["ment", "flow", "nn"]]
            for i, model in enumerate(_models):
                x = model.sample(int(n_samples))
                
                hist, edges = np.histogramdd(x[:, (0, 1)], bins=bins, range=(2 * [(-xmax, xmax)]))
                hist = np.ma.masked_less_equal(hist, 0.0)
                
                axs[i, run_index].pcolormesh(
                    edges[0], 
                    edges[1], 
                    hist.T, 
                    cmap=cmaps[i],
                    **plot_kws
                )

                # Plot projections.
                transforms = models["flow"].transforms
                for meas_index, transform in enumerate(transforms):
                    u = transform(x)
                    u = grab(u)
                        
                    hist, edges = np.histogram(u[:, 0], bins=150, range=(-xmax, xmax), density=True)
                    hist = hist / np.max(hist)
                                        
                    psv.plot_profile(
                        hist,
                        edges=edges,
                        ax=axs[4 + meas_index, run_index],
                        **lineplot_kws[i]
                    )

            ## Draw integration lines.
            angles = np.linspace(cfg.meas.min_angle, cfg.meas.max_angle, cfg.meas.num, endpoint=False)
            angles = np.radians(angles)
            angles = angles + 0.5 * np.pi
            rmax = 1.0 * xmax
            for angle in angles:
                x = rmax * np.cos(angle)
                y = rmax * np.sin(angle)
                for ax in axs[:n_models, run_index]:
                    ax.plot(
                        [-x, x],
                        [-y, y],
                        color="black",
                        lw=0.35,
                        alpha=0.25,
                        zorder=9999,
                    )

            n_empty = max_n_meas - cfg.meas.num
            if n_empty > 0:
                for ax in axs[-n_empty:, run_index]:
                    ax.axis("off")
            
    # Formatting
    limits = 2 * [(-xmax, xmax)]
    axs[:n_models, :].format(xlim=limits[0], ylim=limits[1])
    axs[n_models:, :].format(xlim=limits[0], ylim=(0.00, ymax))
    axs.format(leftlabels=(["True", "MENT", "Flow", "NN"] + [""] * max_n_meas))
    
    return fig, axs


for dist_name in tqdm(dist_names):    
    fig, axs = plot_compare_all_profiles(
        dist_name=dist_name, 
        n_samples=1_000_000,
        xmax=3.0,
        bins=125, 
    )
    
    filename = f"fig_rec_2d_{dist_name}.pdf"
    filename = os.path.join(output_dir, filename)
    print(f"saving file {filename}")
          
    plt.savefig(filename, dpi=100)
    plt.close("all")


