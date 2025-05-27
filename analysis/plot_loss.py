"""Plot training loss."""
import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import ultraplot as uplt

import mentflow as mf
from mentflow.utils import grab
from mentflow.utils import unravel

sys.path.append("..")
from experiments.load import load_mentflow_run


uplt.rc["cmap.discrete"] = False
uplt.rc["cmap.sequential"] = "viridis"
uplt.rc["figure.facecolor"] = "white"
uplt.rc["grid"] = False

uplt.rc["font.family"] = "serif"
uplt.rc["text.usetex"] = True


device = "cpu"
precision = torch.float32
send = lambda x: x.type(precision).to(device)

data_dir = f"../experiments/rec_2d/linear/outputs/train_flow/240723184148/"

run = load_mentflow_run(data_dir, device=device)

cfg = run["config"]
history = run["history"]
model = run["model"]
checkpoints = run["checkpoints"]

colors = ["black", "gray"]
lw = 1.0
lss = ["-", "-"]

fig, ax1 = uplt.subplots(figwidth=3.5, figheight=2.0)
ax1.plot(history["H"], color=colors[0], lw=lw, ls=lss[0], zorder=9999)
ax1.format(
    xlabel="Iteration (global)", 
    ylabel=r"$-H \left[ \rho(\mathbf{x}), \rho_*(\mathbf{x}) \right]$",
    xmin=-200,
)

ax2 = ax1.twinx(color=colors[1])
ax2.plot(history["D_norm"], color=colors[1], lw=lw, ls=lss[1], zorder=0)
ax2.format(ylabel=r"$D \left[ g(\mathbf{u}_{k_\parallel}), \tilde{g}(\mathbf{u}_{k_\parallel}) \right]$")

ax1.spines["left"].set_color(colors[0])
ax1.yaxis.label.set_color(colors[0])
ax1.tick_params(axis="y", colors=colors[0])

with uplt.rc.context({
    "font.family": "sans-serif",
    "text.usetex": False,
}):
    for i, (y, text) in enumerate(zip([0.60, 0.125], ["negative entropy", "data mismatch"])):
        ax1.annotate(
            text, 
            xy=(0.57, y),
            xycoords="axes fraction", 
            color=colors[i], 
            # fontsize=8.0,
            fontfamily="sans-serif",
        )

if not os.path.exists("outputs"):
    os.makedirs("outputs")
    
plt.savefig("./outputs/fig_loss.pdf", dpi=300)



