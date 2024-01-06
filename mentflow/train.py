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
import proplot as pplt
import torch
from tqdm.notebook import tqdm as tqdm_nb
from tqdm import tqdm

from mentflow.core import MENTFlow

from mentflow.diagnostics import Histogram1D
from mentflow.diagnostics import kde_histogram_1d

from mentflow.utils import save_pickle
from mentflow.utils import unravel
from mentflow.utils.logging import ListLogger


class ScriptManager:
    """Helps setup paths, logger, etc.

    Parameters
    ----------
    file_path : str
        Full path to the script file.
    output_dir : str
        Full path to output directory. The directory will be created if it does 
        not exist. Files will be saved to /{output_dir}/{file_path.stem}/{timestamp}. 
        Example:
            - file_path = "/path/to/script_name.py"
            - output_dir = "/path/to/output/"
            - timestamp = "240104215723"
            - output_dir --> "/path/to/output/script_name/240104215723/"
    """
    def __init__(self, file_path: str, output_dir : str) -> None:
        self.datestamp = time.strftime("%Y-%m-%d")
        self.timestamp = time.strftime("%y%m%d%H%M%S")
        self.file_path = pathlib.Path(file_path)
        self.output_dir = os.path.join(output_dir, self.file_path.stem, self.timestamp)
        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)

    def get_path(self, filename : str) -> str:
        """Get full path to output file name."""
        return os.path.join(self.output_dir, filename)

    def make_dirs(self, *dir_names) -> None:
        """Make directories in output folder."""
        for dir_name in dir_names:
            path = os.path.join(self.output_dir, dir_name)
            if not os.path.exists(path):
                os.makedirs(path)

    def save_pickle(self, object, filename):
        """Pickle object and save to output directory."""
        return save_pickle(object, self.get_path(filename))

    def save_script_copy(self):
        """Save a copy of the script file to the output directory."""
        old_path = str(self.file_path)
        new_path = self.get_path(self.file_path.name)
        shutil.copy(old_path, new_path)

    def get_logger(self, disp: bool = True, filename : str = "log.txt") -> logging.Logger:
        """Return logger."""
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

    def update(self, val):
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


class Monitor:
    """Monitors MENT-Flow traing progress."""
    def __init__(
        self, 
        model: MENTFlow, 
        optimizer: Optional[torch.optim.Optimizer] = None, 
        lr_scheduler: Optional[Any] = None,
        momentum: Optional[float] = 0.85,
        path: Optional[str] = None,
        notebook: bool = False
    ) -> None:
        """Constructor."""
        self.logger = ListLogger(save=(path is not None), path=path, freq=1)
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.path = path

        self.meters = {
            "L": RunningAverageMeter(momentum=momentum),
            "H": RunningAverageMeter(momentum=momentum),
            "D": [RunningAverageMeter(momentum=momentum) for _ in unravel(self.model.measurements)],
            "D_norm": RunningAverageMeter(momentum=momentum),
        }

        self.progress_bar = None
        self.notebook = notebook
                
        self.start_time = None
        self.best_loss = float("inf")
        self.best_state_dict = model.state_dict()

    def set_progress_bar(self, iterations: int):
        self.progress_bar = None
        if self.notebook:
            self.progress_bar = tqdm_nb(total=iterations)
        else:
            self.progress_bar = tqdm(total=iterations)
        return self.progress_bar

    def update(
        self, 
        epoch : int, 
        iteration : int, 
        L: float, 
        H: float, 
        D: List[float], 
        batch_size: int
    ) -> None:

        # Update time ellapsed.
        if self.start_time is None:
            self.start_time = time.time()
        time_ellapsed = time.time() - self.start_time

        # Get learning rate.
        lr = self.optimizer.param_groups[0]["lr"]

        # Update average meters.
        self.meters["L"].update(float(L))
        self.meters["H"].update(float(H))
        for i in range(len(D)):
            self.meters["D"][i].update(float(D[i]))
        D_norm = float(sum(D)) / len(D)
        self.meters["D_norm"].update(D_norm)

        # Update best state dict.
        if L < self.best_loss:
            self.best_loss = L
            self.best_state_dict = copy.deepcopy(self.model.state_dict())

        # Write line to output file.
        info = dict()
        info["time"] = time_ellapsed
        info["epoch"] = epoch
        info["iteration"] = iteration
        info["learning_rate"] = lr
        info["batch_size"] = batch_size
        info["penalty"] = self.model.penalty_parameter
        info["L"] = float(L)
        info["H"] = float(H)
        info["D_norm"] = float(D_norm)
        for i in range(len(D)):
            info[f"D_{i:02.0f}"] = float(D[i])
        self.logger.write(info)

        # Update progress bar.
        description = (
            "L={:0.2e}".format(info["L"]),
            "H={:0.2e}".format(info["H"]),
            "|D|={:0.2e}".format(info["D_norm"]),
            "lr={:0.2e}".format(info["learning_rate"]),
            "n={:0.2e}".format(info["batch_size"]),
        )
        description = " ".join(description)
        self.progress_bar.set_description(description)
        self.progress_bar.update()

        return info

    def reset(self) -> None:
        self.best_loss = float("inf")
        self.meters["L"].reset()
        self.meters["H"].reset()
        self.meters["D_norm"].reset()
        for i in range(len(self.meters["D"])):
            self.meters["D"][i].reset()


class Trainer:
    """"MENT-Flow model trainer."""
    def __init__(
        self,
        model: MENTFlow,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: Any,
        plotter=None,
        evaluator=None,
        output_dir=None,
        precision=torch.float32,
        device=None,
        notebook=False,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        
        self.precision = precision
        self.device = device
        self.notebook = notebook

        # Make output directories.
        self.output_dir = output_dir
        if self.output_dir is not None:
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)
    
            self.fig_dir = os.path.join(self.output_dir, f"figures")
            if not os.path.exists(self.fig_dir):
                os.makedirs(self.fig_dir)
                
            self.checkpoint_dir = os.path.join(self.output_dir, f"checkpoints")
            if not os.path.exists(self.checkpoint_dir):
                os.makedirs(self.checkpoint_dir)

        # Setup monitors.
        path = None
        if self.output_dir is not None:
            path = os.path.join(self.output_dir, "history.pkl")
            
        self.monitor = Monitor(
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            momentum=0.85,
            path=path,
            notebook=notebook,
        )
        self.plotter = plotter
        self.evaluator = evaluator

    def _get_filename(self, filename: str, epoch: int, iteration: int, ext: str = None) -> str:
        filename = f"{filename}_{epoch:03.0f}_{iteration:05.0f}"
        if ext is not None:
            filename = f"{filename}.{ext}"
        return filename

    def save_checkpoint(self, epoch: int, iteration: int, state_dict=None) -> None:
        curr_state_dict = self.model.state_dict()
        if state_dict is not None:
            self.model.load_state_dict(state_dict)

        filename = self._get_filename("model", epoch, iteration, ext="pt")
        filename = os.path.join(self.checkpoint_dir, filename)
        print(f"Saving file {filename}")
        self.model.save(filename)

        if state_dict is not None:
            self.model.load_state_dict(curr_state_dict)

    def make_and_save_plots(self, epoch: int, iteration: int, state_dict=None, **savefig_kws) -> None:
        if self.plotter is None:
            return

        curr_state_dict = self.model.state_dict()
        if state_dict is not None:
            self.model.load_state_dict(state_dict)

        ext = savefig_kws.pop("ext", "png")
        figs = self.plotter(self.model)
        for index, fig in enumerate(figs):
            if self.output_dir is not None:
                filename = self._get_filename(f"fig_{index:02.0f}", epoch, iteration, ext=ext)
                filename = os.path.join(self.fig_dir, filename)
                print(f"Saving file {filename}")
                fig.savefig(filename, **savefig_kws)
            if self.notebook:
                plt.show()
            plt.close("all")

        if state_dict is not None:
            self.model.load_state_dict(curr_state_dict)

    def train(
        self,
        epochs: int = 20,
        iterations: int = 1000,
        batch_size: int = 30000,
        rtol: float = 0.05,
        atol: float = 0.0,
        dmax: float = 0.0,
        penalty_step: float = 20.0,
        penalty_scale: float = 1.1,
        penalty_max: float = None,
        vis_freq: int = None,
        eval_freq: int = None,
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
        of the discrepancy vector D, divided by the length of C. Training will cease as soon 
        as |D| <= dmax * len(D). The ideal stopping point is usually clear from a plot of |D| 
        vs. iteration number. Eventually, large increases in H will be required for very 
        small decreases in |D|; we want to stop before this occurs.
        
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
        vis_freq : int
            Visualization frequency.  Defaults to `iterations` (saves after each epoch).
        eval_freq : int
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
            
        if not eval_freq:
            eval_freq = iterations

        
        def train_epoch(epoch):
            """Train one epoch (inner loop)."""
            self.monitor.reset()
            self.monitor.set_progress_bar(iterations)
            
            for iteration in range(iterations):   
                self.optimizer.zero_grad()
                loss, H, D = self.model.loss(batch_size)
    
                info = self.monitor.update(
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
                        self.make_and_save_plots(
                            epoch,
                            iteration,
                            state_dict=self.monitor.best_state_dict,
                            **savefig_kws,
                        )
                    self.model.train()
    
                if ((iteration + 1) % eval_freq == 0) or ((iteration + 1) == iterations):
                    self.model.eval()
                    with torch.no_grad():
                        curr_state_dict = self.model.state_dict()
                        self.model.load_state_dict(self.monitor.best_state_dict)
                        if self.evaluator is not None:
                            self.evaluator(self.model)
                        self.save_checkpoint(epoch, iteration)
                        self.model.load_state_dict(curr_state_dict)
                    self.model.train()
        
                self.lr_scheduler.step(loss.item())

            self.monitor.progress_bar.close()


        # Outer loop: Penalty Method (PM).
        # --------------------------------------------------------------------------
        
        converged = False
        converged_message = ""
        final_epoch = False
        D_norm_old = float("inf")
        
        for epoch in range(epochs):            
            # Print update statement.
            print("epoch = {:}".format(epoch))
            print("penalty = {:}".format(self.model.penalty_parameter))

            # Solve the current subproblem (fixed penalty parameter).
            train_epoch(epoch)

            # Load the best state dict from this subproblem and compute the 
            # discrepancy with 100,000 particles.
            with torch.no_grad():
                self.model.eval()
                current_state_dict = self.model.state_dict()
                self.model.load_state_dict(self.monitor.best_state_dict)
                
                loss, H, D = self.model.loss(batch_size=100000)
                D_norm = float(sum(D)) / len(D)
                
                self.model.load_state_dict(current_state_dict)
                self.model.train()

            # Print change in data fit.
            print("D_norm = {:0.3e}".format(D_norm))
            print("D_norm_old = {:0.3e}".format(D_norm_old))
            print("D_norm_abs_change = {:0.3e}".format(D_norm - D_norm_old))
            print("D_norm_rel_change = {:0.3e}".format(D_norm / D_norm_old))
            print()

            # Check for convergence.
            if D_norm <= dmax:
                converged = True
                converged_message = "CONVERGED (dmax)"
            if D_norm > (1.0 - rtol) * D_norm_old:
                converged = True
                converged_message = "CONVERGED (rtol)"
            if D_norm_old - D_norm < atol:
                converged = True
                converged_message = "CONVERGED (atol)"

            # If converged, train one more epoch with the same penalty parameter. 
            # Otherwise, increase the penalty parameter.
            if converged:
                if final_epoch:
                    self.model.load_state_dict(self.monitor.best_state_dict)
                    return
                print(converged_message)
                print("Training one more epoch with same penalty parameter")
            else:
                self.model.penalty_parameter *= penalty_scale
                self.model.penalty_parameter += penalty_step
                if self.model.penalty_parameter >= penalty_max:
                    print("Max penalty parameter reached.")
                    return   

            # Store variables for the next epoch.
            final_epoch = converged
            D_norm_old = D_norm            
        
        # Load the best state dict from the last epoch.
        self.model.load_state_dict(self.monitor.best_state_dict)
