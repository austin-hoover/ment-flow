"""Make phase space tomography diagram (Fig. 1)."""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import ultraplot as uplt
from ipywidgets import interact
from ipywidgets import widgets
from matplotlib.patches import ConnectionPatch
from matplotlib.patches import Ellipse
from matplotlib.patches import Rectangle


uplt.rc["cmap.discrete"] = False
uplt.rc["cmap.sequential"] = "viridis"
uplt.rc["figure.facecolor"] = "white"
uplt.rc["grid"] = False

uplt.rc["font.family"] = "serif"
uplt.rc["text.usetex"] = True


# Setup
# ----------------------------------------------------------------------------------------

input_filename = "./sns_btf_simulation/bunch_sns-btf_vt36a_normalized.npy"
output_dir = "./outputs"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    

def forward(x):
    angle = 0.25 * np.pi
    M = np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])
    u = np.matmul(x, M.T)
    u[:, 1] += 0.075 * u[:, 0] ** 2
    return u


def inverse(u):
    u = np.copy(u)
    u[:, 1] *= -1.0
    u = forward(u)
    u[:, 1] *= -1.0
    return u


x = np.load(input_filename)
x = x[:, :2]
x = x[:2000]

u = forward(x)


# Figure
# ----------------------------------------------------------------------------------------

# Settings
xmax = 5.0
limits = (-xmax, xmax)
fontsize = 11.5

# Create figure
fig, axs = uplt.subplots(ncols=3, figwidth=4.5, width_ratios=[1.0, 1.0, 1.0])
axs.format(xlim=limits, ylim=limits)
axs.format(xspineloc="neither", yspineloc="neither")

# Create panel axes
paxs = [ax.panel_axes("bottom") for ax in axs]
for pax in paxs[:-1]:
    pax.format(xspineloc="neither", yspineloc="neither")
pax = paxs[-1]
pax.format(xspineloc="bottom", yspineloc="neither", xticks=[])
pax.spines["bottom"].set(linewidth=1.25, color="black")

# Plot initial/transformed coordinates.
axs[0].scatter(x[:, 0], x[:, 1], c="black", ec="None", s=0.75)
axs[2].scatter(u[:, 0], u[:, 1], c="black", ec="None", s=0.75)

# Plot projection onto x axis.
pax.hist(x[:, 0], bins=50, color="black", histtype="step", lw=1.5)

# Draw integration lines
res = 20
scale = 2.0
coords = scale * np.linspace(-xmax, xmax, int(scale * res))
for i in range(len(coords)):
    line = np.zeros((int(scale * 100), 2))
    line[:, 0] = coords[i]
    line[:, 1] = scale * np.linspace(-xmax, xmax, int(scale * 100))
    line1 = line
    line2 = inverse(line)
    for ax, line in zip([axs[2], axs[0]], [line1, line2]):
        ax.plot(line[:, 0], line[:, 1], color="black", alpha=0.1, zorder=0, lw=1.0)    

# Add arrows
line_kws = dict(ls="-", color="black",)

arrow_length = 4.0
axs[1].arrow(
    -arrow_length,
    +0.0,
    +2.0 * arrow_length,
    +0.0,
    head_width=0.35,
    head_length=0.35,
    length_includes_head=True,
    **line_kws
)
connection1 = ConnectionPatch(
    (-0.2, 0.0), 
    ( 0.5, 0.0), 
    coordsA="axes fraction", 
    coordsB="axes fraction",
    axesA=paxs[2],
    axesB=paxs[0], 
    arrowstyle='-', 
    **line_kws
)
connection2 = ConnectionPatch(
    (0.5, 0.0), 
    (0.5, 0.5), 
    coordsA="axes fraction", 
    coordsB="axes fraction",
    axesA=paxs[0],
    axesB=paxs[0], 
    arrowstyle="-|>",
    **line_kws
)
# fig.add_artist(connection1)
# fig.add_artist(connection2)

# Add label $\mathcal{M}_k$
axs[1].annotate(
    # r"$\mathbf{u}_k = \mathcal{M}_k (\mathbf{x})$",
    r"$\mathcal{M}_k$",
    xy=(0.5, 0.625),
    xycoords="axes fraction",
    horizontalalignment="center",
    verticalalignment="center",
    fontsize=fontsize,
)

# # Add label "Accelerator"
# axs[1].annotate(
#     "Accelerator",
#     xy=(0.5, 0.35),
#     xycoords="axes fraction",
#     horizontalalignment="center",
#     verticalalignment="center",
#     fontsize=fontsize,
# )

# # Add label "Reconstruction"
# paxs[1].annotate(
#     "?",
#     xy=(0.5, 0.35), 
#     xycoords="axes fraction",
#     horizontalalignment="center",
#     verticalalignment="center",
#     annotation_clip=False,
#     fontsize=fontsize * 1.2,
# )

# Label axes
axs[0].format(title=r"$\rho(\mathbf{x})$")
axs[2].format(title=r"$\rho(\mathbf{u}_k)$")
pax.format(xlabel=r"$g_k(\mathbf{u}_{k_\parallel})$")
axs.format(title_kw=dict(fontsize=fontsize), xlabel_kw=dict(fontsize=fontsize))


# Save figure
plt.savefig(os.path.join(output_dir, f"fig_diagram.pdf"), dpi=300)













