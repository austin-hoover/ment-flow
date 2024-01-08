"""Training helpers."""
import copy
import os
import time
import typing
from typing import Any
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import proplot as pplt
import torch
from tqdm.notebook import tqdm as tqdm_nb
from tqdm import tqdm

from mentflow.core import MENTFlow
from mentflow.utils.logging import ListLogger


class Trainer:
    """"MENT-Flow model trainer."""
    def __init__(
        self,
        model: MENTFlow,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: Any,
        plot=None,
        eval=None,
        output_dir=None,
        notebook=False,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        self.plot = plot
        self.eval = eval
        
        self.notebook = notebook

        self.output_dir = output_dir
        if self.output_dir is not None:
            os.makedirs(self.output_dir, exist_ok=True)
    
            self.fig_dir = os.path.join(self.output_dir, f"figures")
            os.makedirs(self.fig_dir, exist_ok=True)
                
            self.checkpoint_dir = os.path.join(self.output_dir, f"checkpoints")
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            
    def get_filename(self, filename: str, epoch: int, iteration: int, ext: str = None) -> str:
        filename = f"{filename}_{epoch:03.0f}_{iteration:05.0f}"
        if ext is not None:
            filename = f"{filename}.{ext}"
        return filename
    
    def plot_model(self, epoch: int, iteration: int, **savefig_kws) -> None:
        if self.plot is None:
            return
            
        ext = savefig_kws.pop("ext", "png")
        figs = self.plot(self.model)
        for index, fig in enumerate(figs):
            if self.output_dir is not None:
                path = self.get_filename(f"fig_{index:02.0f}", epoch, iteration, ext=ext)
                path = os.path.join(self.fig_dir, path)
                print(f"Saving file {path}")
                fig.savefig(path, **savefig_kws)
            if self.notebook:
                plt.show()
            plt.close("all")

    def eval_model(self, epoch: int, iteration: int):
        if self.eval is not None:
            self.eval(self.model)

        path = self.get_filename("model", epoch, iteration, ext="pt")
        path = os.path.join(self.checkpoint_dir, path)
        print(f"Saving file {path}")
        self.model.save(path)

    def get_progress_bar(self, length):
        if self.notebook:
            return tqdm_nb(total=length)
        else:
            return tqdm(total=length)

    def train(
        self,
        epochs: int = 20,
        iterations: int = 1000,
        batch_size: int = 30000,
        rtol: float = 0.05,
        atol: float = 0.0,
        dmax: float = 0.0,
        penalty_start: float = 0.0,
        penalty_step: float = 20.0,
        penalty_scale: float = 1.1,
        penalty_max: float = None,
        eval_freq: int = None,
        savefig_kws: Optional[dict] = None,
    ) -> None:
        """Train using the penalty method.
        
        The `penalty_step` and `penalty_scale` parameters determine the penalty parameter step 
        size and scaling factor after each epoch. These numbers should be as small as possible 
        to avoid ill-conditioning.
        
        The `dmax` parameter defines the convergence condition --- the maximum allowed L1 norm
        of the discrepancy vector D, divided by the length of D. Training will cease after 
        |D| <= dmax * len(D). The ideal stopping point is usually clear from a plot of |D|  and
        H vs. iteration number. Eventually, a large increase in H will be required for a small 
        decreases in |D|; we want to stop before this point.
        
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
            Stop if |D| <= dmax.
        penalty_start, penalty_scale, penalty_step, penalty_max: float
            Start, scale, step, and max for penalty parameter.
        eval_freq
            Evaluation frequency.  Defaults to after each epoch.
        savefig_kws : dict
            Key word arguments for matplotlib.savefig. 
        """
        if not savefig_kws:
            savefig_kws = dict()
        savefig_kws.setdefault("dpi", 300)
                        
        if not eval_freq:
            eval_freq = iterations

        if penalty_max is None:
            penalty_max = float("inf")

        path = None
        if self.output_dir is not None:
            path = os.path.join(self.output_dir, "history.pkl")
        logger = ListLogger(path=path)

        start_time = time.time()

        
        def train_epoch(epoch):   
            """Inner loop (fixed penalty parameter)."""
            best_loss = float("inf")
            best_state_dict = self.model.state_dict()
            progress_bar = self.get_progress_bar(iterations)
                            
            for iteration in range(iterations):   
                self.optimizer.zero_grad()
                loss, H, D = self.model.loss(batch_size)

                if not (torch.isinf(loss) or torch.isnan(loss)):
                    loss.backward()
                    self.optimizer.step()
        
                # Log info.
                info = dict()
                info["epoch"] = epoch
                info["iteration"] = iteration
                info["L"] = float(loss)
                info["H"] = float(H)
                info["D_norm"] = float(sum(D) / len(D))
                info["batch_size"] = batch_size
                info["learning_rate"] = self.optimizer.param_groups[0]["lr"]
                info["penalty"] = self.model.penalty_parameter
                info["time"] = time.time() - start_time
                logger.write(info)
        
                # Update progress bar.
                description = (
                    "L={:0.2e}".format(info["L"]),
                    "H={:0.2e}".format(info["H"]),
                    "D={:0.2e}".format(info["D_norm"]),
                    "lr={:0.2e}".format(info["learning_rate"]),
                    "n={:0.2e}".format(info["batch_size"]),
                )
                description = " ".join(description)
                progress_bar.set_description(description)
                progress_bar.update()

                # Update best state dict
                if loss < best_loss:
                    best_loss = loss
                    best_state_dict = copy.deepcopy(self.model.state_dict())

                # Evaluate
                if ((iteration + 1) % eval_freq == 0) or ((iteration + 1) == iterations):
                    self.model.eval()
                    with torch.no_grad():
                        curr_state_dict = self.model.state_dict()
                        self.model.load_state_dict(best_state_dict)
                        self.plot_model(epoch, iteration, **savefig_kws)
                        self.eval_model(epoch, iteration)                        
                        self.model.load_state_dict(curr_state_dict)
                    self.model.train()

                # Scale learning rate
                self.lr_scheduler.step(loss.item())

            progress_bar.close()
            return best_state_dict


        # Outer loop (penalty method)        
        converged = False
        converged_message = ""
        final_epoch = False
        D_norm_old = float("inf")
        self.model.penalty_parameter = penalty_start
        
        for epoch in range(epochs):            
            print("epoch = {:}".format(epoch))
            print("penalty = {:}".format(self.model.penalty_parameter))

            # Solve the current subproblem (fixed penalty parameter).
            best_state_dict = train_epoch(epoch)

            # Load the best state dict and compute the discrepancy with 100,000 particles.
            with torch.no_grad():
                self.model.eval()
                current_state_dict = self.model.state_dict()
                self.model.load_state_dict(best_state_dict)
                
                loss, H, D = self.model.loss(batch_size=100000)
                D_norm = float(sum(D)) / len(D)
                
                self.model.load_state_dict(current_state_dict)
                self.model.train()

            # Print the change in discrepancy.
            print("D_norm = {:0.3e}".format(D_norm))
            print("D_norm_old = {:0.3e}".format(D_norm_old))
            print("D_norm_abs_change = {:0.3e}".format(D_norm - D_norm_old))
            print("D_norm_rel_change = {:0.3e}".format(D_norm_old / D_norm))
            print()

            # Check if convergence.
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
                    self.model.load_state_dict(best_state_dict)
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
        
        # Before leaving, load the best state dict from the last epoch.
        self.model.load_state_dict(best_state_dict)
