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

from mentflow.core import MENTModel
from mentflow.utils.logging import ListLogger
import mentflow.losses


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
        filename = self.get_filename(filename)
        
        info_file_handler = logging.FileHandler(filename, mode="a")
        info_file_handler.setLevel(logging.INFO)
        logger.addHandler(info_file_handler)
        
        if disp:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            logger.addHandler(console_handler)
            
        return logger

    def get_filename(self, filename : str) -> str:
        return os.path.join(self.outdir, filename)

    def make_folders(self, *names) -> None:
        for name in names:
            path = os.path.join(self.outdir, name)
            if not os.path.exists(path):
                os.makedirs(path)


class Monitor:
    """Monitors MENT-Flow traing progress."""
    def __init__(
        self, 
        model: MENTModel, 
        optimizer: Optional[torch.optim.Optimizer] = None, 
        lr_scheduler: Optional[Any] = None,
        momentum: Optional[float] = 0.99,
        path: Optional[str] = None,
        freq : int = 1,
    ) -> None:
        self.path = path
        self.logger = ListLogger(save=(path is not None), path=path, freq=freq)

        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.n_meas = len(model.measurements)
        self.meters = {
            "L": RunningAverageMeter(momentum=momentum),
            "H": RunningAverageMeter(momentum=momentum),
            "C": [RunningAverageMeter(momentum=momentum) for _ in range(self.n_meas)],
            "C_norm": RunningAverageMeter(momentum=momentum),
        }
                
        self.start_time = None
        self.best_loss = float("inf")
        self.best_state_dict = model.state_dict()

    def action(
        self, 
        step : int, 
        iteration : int, 
        L: float, 
        H: float, 
        C: List[float], 
        batch_size: int
    ) -> None:
        """Update history array."""
        if self.start_time is None:
            self.start_time = time.time()
        time_ellapsed = time.time() - self.start_time

        lr = self.optimizer.param_groups[0]["lr"]

        self.meters["L"].action(float(L))
        self.meters["H"].action(float(H))

        for i in range(self.n_meas):
            self.meters["C"][i].action(float(C[i]))
            
        C_norm = sum(float(C[i])**2 for i in range(len(C))) ** 0.5
        self.meters["C_norm"].action(C_norm)

        if L < self.best_loss:
            self.best_loss = L
            self.best_state_dict = copy.deepcopy(self.model.state_dict())

        info = dict()
        info["step"] = step
        info["iteration"] = iteration
        info["t"] = time_ellapsed
        info["batch_size"] = batch_size
        info["lr"] = lr
        info["L"] = float(L)
        info["H"] = float(H)
        info["C_norm"] = float(C_norm)
        info["mu"] = self.model.penalty_parameter
        for i in range(self.n_meas):
            info[f"C_{i:02.0f}"] = float(C[i])
        for i in range(len(self.model.lagrange_multipliers)):
            info[f"lambd_{i:02.0f}"] = self.model.lagrange_multipliers[i]
        self.logger.write(info)

        message = "step={:02.0f} iter={:05.0f} t={:0.2f} L={:0.2e} H={:0.2e} C={:0.2e} lr={} batch={:0.2e}".format(
            step,
            iteration,
            time_ellapsed,
            float(L),
            float(H),
            float(C_norm),
            lr,
            batch_size,
        )
        print(message)

    def reset(self) -> None:
        self.meters["L"].reset()
        self.meters["H"].reset()
        self.meters["C_norm"].reset()
        for i in range(self.n_meas):
            self.meters["C"][i].reset()
        self.best_loss = float("inf")


class Trainer:
    """"Augmented Lagrangian training for MENT-Flow model."""
    def __init__(
        self,
        model: MENTModel,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: Any,
        monitor: Monitor,
        plotter: Callable[[MENTModel], List[plt.Figure]],
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

    def get_prefix(self, step: int, iteration: int) -> str:
        return f"{step:03.0f}_{iteration:05.0f}"

    def save_checkpoint(self, step: int, iteration: int) -> None:
        filename = f"model_{self.get_prefix(step, iteration)}.pt"
        filename = os.path.join(self.checkpoint_dir, filename)
        print(f"Saving file {filename}")
        self.model.save(filename)

    def make_and_save_plots(self, step: int, iteration: int, save=False, **savefig_kws) -> None:
        if self.plotter is None:
            return
            
        ext = savefig_kws.pop("ext", "png")

        figures = self.plotter(self.model)

        for index, figure in enumerate(figures):
            if save:
                filename = f"fig_{index:02.0f}_{self.get_prefix(step, iteration)}.{ext}"
                filename = os.path.join(self.fig_dir, filename)
                print(f"Saving file {filename}")
                figure.savefig(filename, **savefig_kws)
            else:
                plt.show()
            plt.close("all")

    def train(
        self,
        steps: int = 20,
        iterations: int = 1000,
        batch_size: int = 30000,
        rtol: float = 0.8,
        penalty_parameter_scale: float = 2.0,
        penalty_parameter_max: float = None,
        save: bool = True,
        vis_freq: int = 100,
        checkpoint_freq: int = 100,
        savefig_kws: Optional[dict] = None,
    ) -> None:
        if savefig_kws is None:
            savefig_kws = dict()
        savefig_kws.setdefault("dpi", 300)

        if penalty_parameter_max is None:
            penalty_parameter_max = float("inf")
            

        def train_step(step):
            print("step={}".format(step))
            print("mu={}".format(self.model.penalty_parameter))
            print("lambd={}".format(self.model.lagrange_multipliers))
            self.monitor.reset()

            for iteration in range(iterations):                
                self.optimizer.zero_grad()
                loss, H, C = self.model.loss(batch_size)
    
                self.monitor.action(
                    step=step,
                    iteration=iteration,
                    L=loss,
                    H=H,
                    C=C,
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
                            step=step,
                            iteration=iteration,
                            save=save,
                            **savefig_kws,
                        )
                        self.model.load_state_dict(curr_state_dict)
                    self.model.train()
    
                if ((iteration + 1) % checkpoint_freq == 0) or ((checkpoint_freq + 1) == iterations):
                    filename = f"model_{self.get_prefix(step=step, iteration=iteration)}.pt"
                    self.model.save(os.path.join(self.checkpoint_dir, filename))
    
                self.lr_scheduler.step(float(loss))
        

        C_norm_old = float("inf")
        n_steps_improved = 0
    
        for step in range(steps):
            train_step(step)
            
            C_norm = self.monitor.meters["C_norm"].avg  
            improved = C_norm < rtol * C_norm_old
            
            if improved:
                # Update the lagrange multipliers.
                C = [meter.avg for meter in self.monitor.meters["C"]]
                for i in range(len(self.model.lagrange_multipliers)):
                    self.model.lagrange_multipliers[i] += self.model.penalty_parameter * C[i]
                n_steps_improved += 1
            else:
                # Stop if increasing the penalty parameter didn't help. This should
                # let us avoid tuning the maximum penalty parameter. It seems to be
                # working, but need to think if there is a better way.
                if step > 0 and n_steps_improved == 0:
                    penalty_parameter_max = self.model.penalty_parameter
                    print("Increasing penalty parameter did not help.")
                    print("Setting penalty_parameter_max = {}".format(penalty_parameter_max))
                    print("You can probably stop training now.")

                # Update the penalty parameter.
                self.model.penalty_parameter = min(
                    penalty_parameter_max,
                    penalty_parameter_scale * self.model.penalty_parameter,
                )
                n_steps_improved = 0
                
            C_norm_old = C_norm

