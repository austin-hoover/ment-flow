import matplotlib.pyplot as plt
import numpy as np
import proplot as pplt
import torch

import mentflow as mf
from mentflow.utils import grab
from mentflow.utils import unravel

from experiments.plotting import plot_image
from experiments.plotting import plot_hist
from experiments.plotting import plot_points
from experiments.plotting import plot_proj_1d as plot_proj


def plot_dist(x0, x, prob=None, coords=None, bins=75, limits=None, **kws):
    ncols = 3
    if prob is None:
        ncols = 2
    fig, axs = pplt.subplots(
        ncols=ncols, xspineloc="neither", yspineloc="neither", space=0.0, share=False
    )

    plot_points(x0, bins=bins, limits=limits, ax=axs[0], **kws)
    plot_points(x, bins=bins, limits=limits, ax=axs[1], **kws)
    if ncols == 3:
        plot_image(prob, coords=coords, ax=axs[2], **kws)
    axs.format(xlim=limits[0], ylim=limits[1])
    return fig, axs


def plot_model(
    model,
    dist,
    n=1000000,
    sim_kws=None,
    bins=100,
    xmax=3.5,
    maxcols=7,
    kind="line",
    colors=None,
    device=None,
):        
    """Plot model during training."""
    if sim_kws is None:
        sim_kws = dict()
    if device is None:
        device = torch.device("cpu")

    def send(x):
        return x.type(torch.float32).to(device)

    # Generate particles.
    x = model.sample(n)
    x = send(x)

    x_true = dist.sample(n)
    x_true = send(x_true)

    # Simulate measurements.
    if type(model) is mf.models.ment.MENT:
        predictions = model.simulate(**sim_kws)
    else:
        kde = [diagnostic.kde for diagnostic in model.diagnostics]
        
        for diagnostic in model.diagnostics:
            diagnostic.kde = False
    
        predictions = model.simulate(x, **sim_kws)
    
        for j, diagnostic in enumerate(model.diagnostics):
            diagnostic.kde = kde[j]

    
    figs = []

    # Plot model vs. true samples.
    fig, axs = plot_dist(
        grab(x_true),
        grab(x),
        bins=bins,
        limits=(2 * [(-xmax, xmax)]),
    )
    figs.append(fig)

    # Plot measured vs. simulated projections.
    fig, axs = plot_proj(
        measurements=[grab(measurement) for measurement in unravel(model.measurements)],
        predictions=[grab(prediction) for prediction in unravel(predictions)],
        bin_edges=grab(model.diagnostics[0].bin_edges),
        maxcols=maxcols,
        kind=kind,
        height=1.25,
        lw=1.5,
    )
    # Plot noiseless histogram of true data samples as faint gray line.
    counter = 0
    for index, transform in enumerate(model.transforms):
        u_true = transform(x_true)
        u_true = grab(u_true)
        edges = grab(model.diagnostics[0].bin_edges)
        for measurement in model.measurements[index]:
            hist, _ = np.histogram(u_true[:, 0], bins=edges, density=True)
            hist = hist / grab(measurement).max()
            plot_hist(
                hist,
                edges,
                ax=axs[counter],
                kind="line",
                color="black",
                alpha=0.3,
                zorder=0,
            )
            counter += 1
    figs.append(fig)

    return figs
