"""Analyze 6:1 reconstructions.

This script outputs three plots per training run. 
1. Measured vs. simulated 1D projections
2. Corner plot (upper diagonal = true, lower diagonal = model)
3. 2D projection (z-z') within 4D sphere in other dimensions (x-x'-y-y')
"""
import os
import pathlib
import pickle
import sys

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import psdist as ps
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
from experiments.load import list_contents
from experiments.load import load_mentflow_run
from experiments.rec_nd_1d.setup import make_distribution


uplt.rc["cmap.discrete"] = False
uplt.rc["cmap.sequential"] = "viridis"
uplt.rc["figure.facecolor"] = "white"
uplt.rc["grid"] = False
uplt.rc["font.family"] = "serif"
uplt.rc["text.usetex"] = True


# Settings
# ----------------------------------------------------------------------------------------

run_dirs = (
    list_contents("../experiments/rec_nd_1d/outputs/train_nn") + 
    list_contents("../experiments/rec_nd_1d/outputs/train_flow")
)

path = pathlib.Path(__file__)

output_dir = "./outputs"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

device = torch.device("cpu")
precision = torch.float32
send = lambda x: x.type(precision).to(device)

n_samples = 1_000_000  # number of samples
xmax = 3.5  # plot limits
bins = 85
mask = False

cmaps = [
    cubehelix_cmap(dark=0.30, color="red"),
    cubehelix_cmap(dark=0.30, color="blue"),
]

colors = [
    "rose",
    "blue9",
]


def get_filename(
    fig_name: str, 
    model_name: str,
    dist_name: str,
    n_meas: int,
    checkpoint_index: int, 
    ext="pdf",
) -> str:
    return f"fig_rec_nd_1d_{model_name}_{dist_name}_{n_meas}-proj_{fig_name}_{checkpoint_index:02.0f}.{ext}"
    

# Make figures
# --------------------------------------------------------------------------------------

for run_dir in run_dirs:        
    print(f"Loading run {run_dir}")
        
    run = load_mentflow_run(run_dir, device=device)
    cfg = run["config"]
    history = run["history"]
    model = run["model"]
    checkpoints = run["checkpoints"]

    ndim = cfg.ndim
    dist_name = cfg.dist.name
    n_meas = cfg.meas.num
    
    model_name = "flow"
    if cfg.gen.name == "nn":
        model_name = "nn"
    
    distribution = make_distribution(cfg)
    
    
    with torch.no_grad():
        for checkpoint_index, checkpoint in enumerate(checkpoints):   
            if checkpoint_index != len(checkpoints) - 1:
                continue

            model.load(checkpoint["path"], device)            
            x1 = grab(distribution.sample(n_samples))
            x2 = grab(model.sample(n_samples))

            
            # Corner plot
            # ----------------------------------------------------------------------------------
            
            with uplt.rc.context(fontsmallsize="xx-large"):                
                grid = psv.CornerGrid(ndim=cfg.ndim, corner=False, figwidth=6.5, space=1.0)
                for i, x in enumerate([x1, x2]):
                    grid.plot(
                        x,
                        lower=i,
                        upper=(not i),
                        diag_kws=dict(color=colors[i], lw=[1.50, 1.20][i], kind="step"),
                        limits=[(-xmax, xmax)] * ndim, 
                        bins=bins, 
                        mask=mask,
                        cmap=cmaps[i],
                    )
                grid.set_labels([r"$x$", r"$x'$", r"$y$", r"$y'$", r"$z$", r"$z'$"])
                grid.axs.format(xticks=[], yticks=[])

                filename = get_filename("corner", model_name, dist_name, n_meas, checkpoint_index, ext="pdf")
                filename = os.path.join(output_dir, filename)
                print(f"Saving file {filename}")
                plt.savefig(filename, dpi=300)
                plt.close("all")
    
    
            # Simulated measurements
            # ----------------------------------------------------------------------------------
            
            maxcols = 9
            maxrows = 11
            figwidth = 4.5
                
            x = send(model.sample(n_samples))
        
            for diagnostic in unravel(model.diagnostics):
                diagnostic.kde = False
        
            predictions = mf.simulate.forward(x, model.transforms, model.diagnostics)
        
            y_meas = [grab(meas) for meas in unravel(model.measurements)]
            y_pred = [grab(pred) for pred in unravel(predictions)]
            edges = [grab(diag.edges) for diag in unravel(model.diagnostics)]
            coords = [grab(diag.coords) for diag in unravel(model.diagnostics)]
        
            ncols = min(len(y_meas), maxcols)
            nrows = int(np.ceil(len(y_meas) / ncols))
            nrows = min(nrows, 11 if cfg.dist.name == "gaussian_mixture" else maxrows)
        
            fig, axs = uplt.subplots(
                ncols=ncols, 
                nrows=nrows, 
                figheight=(0.33 * nrows + 0.30), 
                figwidth=figwidth,
                space=0.0,
                xticks=[],
                yticks=[],
            )
            for index in range(len(y_meas)):
                if index < len(axs):
                    ax = axs[index]
                    scale = np.max(y_meas[index])
                    ax.plot(coords[index], y_meas[index] / scale, color=colors[0], lw=1.25, ls="-")
                    ax.plot(coords[index], y_pred[index] / scale, color=colors[1], lw=0.90, ls="-")
            axs.format(ymax=1.25, ymin=0.0)
            axs.format(suptitle=model_name.upper(), suptitle_kw=dict(fontsize="x-large", fontweight="bold"))

            filename = get_filename("sim", model_name, dist_name, n_meas, checkpoint_index, ext="pdf")
            filename = os.path.join(output_dir, filename)
            print(f"Saving file {filename}")
            plt.savefig(filename, dpi=300)
            plt.close("all")


            # Spherical slice (with diagram)
            # ----------------------------------------------------------------------------------
            
            axis_view = (0, 1)  # projection axis
            batch_size = 1_000_000
            n_batches = 10
            limits = 2 * [(-xmax, xmax)]  # plot limits
            bins = 65  # plot histogram bins
            ncols = 4  # number of slice radii 
            
            ndim = cfg.ndim
            axis_slice = tuple([k for k in range(ndim) if k not in axis_view])
            slice_radii = np.linspace(3.0, 1.0, ncols)
                    
            layout = [
                [1, 1, 2, 2, 3, 3, 4, 4],
                [5, 5, 6, 6, 7, 7, 8, 8],
                [None, None, None, 9, 9, None, None, None],
            ]
                    
            fig, axs = uplt.subplots(
                layout,
                figwidth=4.5,
                hspace=[0.0, 3.0],
                wspace=0.0,
            )
            index = 0
            for i, _dist in enumerate([distribution,  model]):
                for j, slice_radius in enumerate(slice_radii):                    
                    hist = np.zeros((bins, bins))
                    edges = 2 * [np.linspace(-xmax, xmax, bins + 1)]
                    for _ in trange(n_batches):
                        x = _dist.sample(batch_size)
                        x = grab(x)
                        x = ps.slice_sphere(x, axis=axis_slice, rmin=0.0, rmax=slice_radius)
                        hist_batch, _ = np.histogramdd(x[:, axis_view], edges)
                        hist += hist_batch
                    hist = hist / np.sum(hist)
                    hist = np.ma.masked_less_equal(hist, 0.0)
                    axs[index].pcolormesh(edges[0], edges[1], hist.T, cmap=cmaps[i])
                    index += 1
            
            dims = [r"$x$", r"$x'$", r"$y$", r"$y'$", r"$z$", r"$z'$"]
            dims_view  = [dims[k] for k in axis_view]
            dims_slice = [dims[k] for k in axis_slice]
            xlabel = dims_view[0]
            ylabel = dims_view[1]
            suptitle = r"$\rho(z, z' \vert r_{\perp} < \tilde{r}_{\perp})$"

            fontsize = "large"
            axs.format(
                xticks=[], 
                yticks=[],
                xlabel=r"$z$", 
                ylabel=r"$z'$",
                suptitle=suptitle, 
                rightlabels=["True", "Model"], 
                xlabel_kw=dict(fontsize=fontsize),
                ylabel_kw=dict(fontsize=fontsize),
                title_kw=dict(fontsize="large"),
                suptitle_kw=dict(fontsize=fontsize, fontweight="normal"),
                rightlabels_kw=dict(rotation=-90),
            )
            
            ax = axs[-1]    
            rmax = 1.2 * slice_radii[0]
            ax.format(
                xlim=(-rmax, rmax), 
                ylim=(-rmax, rmax),
                xspineloc="bottom", 
                yspineloc="left", 
                xlabel=r"$x$-$y$", 
                ylabel=r"$x'$-$y'$",
                xlabel_kw=dict(fontsize=fontsize),
                ylabel_kw=dict(fontsize=fontsize),
            )
            
            for index, radius in enumerate(slice_radii):
                ax.add_patch(patches.Circle((0.0, 0.0), radius=radius, ec="black", lw=1.0, fill=False))
                ax.add_patch(patches.Circle((0.0, 0.0), radius=radius, lw=0.0, fc="black", alpha=0.0))
                theta = 0.5 * np.pi
                theta = theta - np.pi * np.linspace(-0.1, 0.1, len(slice_radii))[index]
                x = radius * np.cos(theta)
                y = radius * np.sin(theta)
                fig.add_artist(
                    patches.ConnectionPatch(
                        xyA=(x, y), 
                        xyB=(0.0, axs[1, index].get_ylim()[0]),
                        coordsA=ax.transData,  
                        coordsB=axs[index + ncols].transData,
                        color="black",
                        lw=0.5,
                        ls="-",
                    )
                )
    
            filename = get_filename("slice", model_name, dist_name, n_meas, checkpoint_index, ext="pdf")
            filename = os.path.join(output_dir, filename)
            print(f"Saving file {filename}")
            plt.savefig(filename, dpi=300)
            plt.close("all")

