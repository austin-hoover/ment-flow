from typing import Callable

from mentflow.core import MENTFlow
from mentflow.types_ import Distribution


def setup_plotter_2d(model: Model, dist: Distribution) -> Callable:
    def plotter(model):
        figs = plotting.plot_model(
            model,
            dist,
            n=cfg.train.vis_size,
            bins=cfg.train.vis_bins,
            xmax=xmax,
            maxcols=cfg.train.vis_maxcols, 
            kind=cfg.train.vis_line,
            device=device
        )