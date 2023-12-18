"""Training helpers."""
import copy
import logging
import math
import os
import pathlib
import pickle
import shutil
import sys
import time
from typing import Any
from typing import Iterable
from typing import Callable
from typing import List
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import proplot as pplt
import torch

from mentflow.core import MENTFlow

from mentflow.diagnostics import Histogram1D
from mentflow.diagnostics import kde_histogram_1d

from mentflow.utils import unravel
from mentflow.utils.logging import ListLogger


class RunningAverageMeter():
    """Computes the exponential moving average."""
    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0
        self.sum = 0
        self.min_avg = float("inf")
        self.n_bad = 0

    def action(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1.0 - self.momentum)
        self.sum += val
        self.val = val
        if self.avg < self.min_avg:
            self.min_avg = self.avg
            self.n_bad = 0
        else:
            self.n_bad += 1


class ScriptManager:
    """Helps setup paths, logger, etc."""
    def __init__(self, filepath: str, outdir : str) -> None:
        self.datestamp = time.strftime("%Y-%m-%d")
        self.timestamp = time.strftime("%y%m%d%H%M%S")
        self.path = pathlib.Path(filepath)
        self.outdir = os.path.join(outdir, self.timestamp)
        if not os.path.isdir(self.outdir):
            os.makedirs(self.outdir)

    def get_logger(self, disp: bool = True, filename : str = "log.txt") -> logging.Logger:
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        path = self.get_path(filename)
        
        info_file_handler = logging.FileHandler(path, mode="a")
        info_file_handler.setLevel(logging.INFO)
        logger.addHandler(info_file_handler)
        
        if disp:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            logger.addHandler(console_handler)
            
        return logger

    def get_path(self, filename : str) -> str:
        return os.path.join(self.outdir, filename)

    def make_dirs(self, *names) -> None:
        for name in names:
            path = os.path.join(self.outdir, name)
            if not os.path.exists(path):
                os.makedirs(path)


class Monitor:
    """Monitors MENT-Flow traing progress."""
    def __init__(
        self, 
        model: MENTFlow, 
        optimizer: Optional[torch.optim.Optimizer] = None, 
        lr_scheduler: Optional[Any] = None,
        momentum: Optional[float] = 0.95,
        path: Optional[str] = None,
        freq : int = 1,
    ) -> None:
        self.path = path
        self.logger = ListLogger(save=(path is not None), path=path, freq=freq)
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.meters = {
            "L": RunningAverageMeter(momentum=momentum),
            "H": RunningAverageMeter(momentum=momentum),
            "D": [RunningAverageMeter(momentum=momentum) for _ in unravel(self.model.measurements)],
            "D_norm": RunningAverageMeter(momentum=momentum),
        }
                
        self.start_time = None
        self.best_loss = float("inf")
        self.best_state_dict = model.state_dict()

    def action(
        self, 
        epoch : int, 
        iteration : int, 
        L: float, 
        H: float, 
        D: List[float], 
        batch_size: int
    ) -> None:
        """Update history array."""
        if self.start_time is None:
            self.start_time = time.time()
        time_ellapsed = time.time() - self.start_time

        lr = self.optimizer.param_groups[0]["lr"]

        self.meters["L"].action(float(L))
        self.meters["H"].action(float(H))
        for i in range(len(D)):
            self.meters["D"][i].action(float(D[i]))
        D_norm = sum(abs(float(D[i])) for i in range(len(D))) / len(D)
        self.meters["D_norm"].action(D_norm)

        if L < self.best_loss:
            self.best_loss = L
            self.best_state_dict = copy.deepcopy(self.model.state_dict())

        info = dict()
        info["epoch"] = epoch
        info["iteration"] = iteration
        info["t"] = time_ellapsed
        info["batch_size"] = batch_size
        info["lr"] = lr
        info["L"] = float(L)
        info["H"] = float(H)
        info["D_norm"] = float(D_norm)
        info["mu"] = self.model.penalty_parameter
        for i in range(len(D)):
            info[f"D_{i:02.0f}"] = float(D[i])
        self.logger.write(info)

        message = "epoch={:02.0f} iter={:05.0f} t={:0.2f} L={:0.2e} H={:0.2e} C={:0.2e} lr={} batch={:0.2e}".format(
            epoch,
            iteration,
            time_ellapsed,
            float(L),
            float(H),
            float(D_norm),
            lr,
            batch_size,
        )
        print(message)

    def reset(self) -> None:
        self.meters["L"].reset()
        self.meters["H"].reset()
        self.meters["D_norm"].reset()
        for i in range(len(self.meters["D"])):
            self.meters["D"][i].reset()
        self.best_loss = float("inf")


class Trainer:
    """"Trainer for MENT-Flow model."""
    def __init__(
        self,
        model: MENTFlow,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: Any,
        monitor: Monitor,
        plotter: Callable[[MENTFlow], List[plt.Figure]],
        save: bool = True,
        output_dir: Optional[str] = None,
        precision=torch.float32,
        device=None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.monitor = monitor
        self.plotter = plotter
        self.device = device
        self.precision = precision
        self.save = save
        
        self.monitor.optimizer = optimizer
        self.monitor.lr_scheduler = lr_scheduler        

        if self.save:
            self.output_dir = output_dir
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)
    
            self.fig_dir = os.path.join(self.output_dir, f"figures")
            if not os.path.exists(self.fig_dir):
                os.makedirs(self.fig_dir)
                
            self.checkpoint_dir = os.path.join(self.output_dir, f"checkpoints")
            if not os.path.exists(self.checkpoint_dir):
                os.makedirs(self.checkpoint_dir)

    def get_prefix(self, epoch: int, iteration: int) -> str:
        return f"{epoch:03.0f}_{iteration:05.0f}"

    def save_checkpoint(self, epoch: int, iteration: int) -> None:
        filename = f"model_{self.get_prefix(epoch, iteration)}.pt"
        filename = os.path.join(self.checkpoint_dir, filename)
        print(f"Saving file {filename}")
        self.model.save(filename)

    def make_and_save_plots(self, epoch: int, iteration: int, save=False, **savefig_kws) -> None:
        if self.plotter is None:
            return
        ext = savefig_kws.pop("ext", "png")
        figures = self.plotter(self.model)
        for index, figure in enumerate(figures):
            if save:
                filename = f"fig_{index:02.0f}_{self.get_prefix(epoch, iteration)}.{ext}"
                filename = os.path.join(self.fig_dir, filename)
                print(f"Saving file {filename}")
                figure.savefig(filename, **savefig_kws)
            else:
                plt.show()
            plt.close("all")

    def train(
        self,
        epochs: int = 20,
        iterations: int = 1000,
        batch_size: int = 30000,
        rtol: float = 0.05,
        atol: float = 0.0,
        dmax: float = 0.0,
        penalty_step: float = 10.0,
        penalty_scale: float = 1.0,
        penalty_max: float = None,
        save: bool = True,
        vis_freq: int = 100,
        checkpoint_freq: int = 100,
        savefig_kws: Optional[dict] = None,
    ) -> None:
        """Train using the Penalty Method (PM).

        For a given lattice, initial distribution, measurement type, and discrepancy function 
        (mean absolute value, KL divergence, etc.), a couple setup runs are usually required 
        to tune the following parameters (in addition to the flow and optimizer parameters).
        
        The `penalty_step` and `penalty_scale` parameters determine the penalty parameter step 
        size and scaling factor after each epoch. These numbers should be as small as possible 
        to avoid ill-conditioning. A reasonable choice is to converge in 10-20 epochs.
        
        The `dmax` parameter defines the convergence condition --- the maximum allowed L1 norm
        of the discrepancy vector C, divided by the length of C. Training will cease as soon 
        as |D| <= dmax * len(D). The ideal stopping point is usually clear from a plot of |C| 
        vs. iteration number. Eventually, large increases in H will be required for very 
        small decreases in |C|; we want to stop before this occurs.
        
        Eventually, an automated stopping condition based on the change in C and H may be
        implemented.

        Parameters
        ----------
        epochs : int
            Number of outer iterations, i.e., penalty parameter updates.
        iterations : int
            Number of ADAM iterations per epoch.
        batch_size : int
            Number of particles sampled on each forward 
        rtol : float
            Stop if |D| > (1 - rtol) * |D_old|. Default = 0.05.
        atol : float
            Stop if |D_old| - |D| < atol. Default = 0 (no progress).
        dmax : float
            Stop if |C| <= dmax. The default is zero (no noise). Note that even in 
            simulated reconstructions without measurement noise, noise can be introduced
            by the particle binning process.
        penalty_scale : float
            Scales the penalty parameter.
        penalty_step: float
            Steps the penalty parameter.
        penalty_max : float
            Maximum penalty parameter value.
        save : bool
            Whether to save plots and checkpoints.
        vis_freq : int
            Visualization frequency.  Defaults to `iterations` (saves after each epoch).
        checkpoint_freq : int
            Checkpoint save frequency. Defaults to `iterations` (saves after each epoch).
        savefig_kws : dict
            Key word arguments for matplotlib.savefig. 
        """
        if savefig_kws is None:
            savefig_kws = dict()
        savefig_kws.setdefault("dpi", 300)

        if penalty_max is None:
            penalty_max = float("inf")
            
        if not vis_freq:
            vis_freq = iterations
            
        if not checkpoint_freq:
            checkpoint_freq = iterations

        
        def train_epoch(epoch):
            """Train one epoch (inner loop)."""
            self.monitor.reset()
            for iteration in range(iterations):                
                self.optimizer.zero_grad()
                loss, H, D = self.model.loss(batch_size)
    
                self.monitor.action(
                    epoch=epoch,
                    iteration=iteration,
                    L=loss,
                    H=H,
                    D=D,
                    batch_size=batch_size,
                )
                
                if not (torch.isinf(loss) or torch.isnan(loss)):
                    loss.backward()
                    self.optimizer.step()

                if ((iteration + 1) % vis_freq == 0) or ((iteration + 1) == iterations):
                    self.model.eval()
                    with torch.no_grad():
                        curr_state_dict = self.model.state_dict()
                        self.model.load_state_dict(self.monitor.best_state_dict)
                        self.make_and_save_plots(
                            epoch=epoch,
                            iteration=iteration,
                            save=save,
                            **savefig_kws,
                        )
                        self.model.load_state_dict(curr_state_dict)
                    self.model.train()
    
                if ((iteration + 1) % checkpoint_freq == 0) or ((iteration + 1) == iterations):
                    if save:
                        curr_state_dict = self.model.state_dict()
                        self.model.load_state_dict(self.monitor.best_state_dict)
                        filename = f"model_{self.get_prefix(epoch=epoch, iteration=iteration)}.pt"
                        self.model.save(os.path.join(self.checkpoint_dir, filename))
                        self.model.load_state_dict(curr_state_dict)

                ## To do: compute T = 0.5 * | grad(D) / |grad(D)| - grad(D) / |grad(H)| |^2 = 0.
                ## T = 0 when entropy is maximized.
                # ...
        
                self.lr_scheduler.step(loss.item())


        # Outer loop: Penalty Method (PM).
        
        D_norm_old = float("inf")
        
        for epoch in range(epochs):
            print("epoch={}".format(epoch))
            print("mu={}".format(self.model.penalty_parameter))

            train_epoch(epoch)
            
            D_norm = self.monitor.meters["D_norm"].avg  
            
            print("D_norm={}".format(D_norm))
            print("D_norm_old={}".format(D_norm_old))
            print("frac={}".format(D_norm / D_norm_old))
            print("diff={}".format(D_norm - D_norm_old))

            converged = False
            if D_norm > (1.0 - rtol) * D_norm_old:
                print("CONVERGED (rtol)")
                return
            if D_norm_old - D_norm < atol:
                print("CONVERGED (atol)")
                return
            if D_norm <= dmax:
                print("CONVERGED (dmax)")
                return 

            self.model.penalty_parameter *= penalty_scale
            self.model.penalty_parameter += penalty_step
            if self.model.penalty_parameter >= penalty_max:
                print("Max penalty parameter reached.")
                return
            
            D_norm_old = D_norm
