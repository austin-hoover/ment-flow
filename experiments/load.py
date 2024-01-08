"""Tools for loading models."""
import os
import pickle
import sys
import typing
from typing import Callable

import torch
import torch.nn as nn
import zuko
from omegaconf import DictConfig

import mentflow as mf


def load_pickle_safe(path):
    """Load pickled file; return None if file does not exist."""
    if os.path.exists(path):
        return mf.utils.load_pickle(path)
    return None


def get_epoch_and_iteration_number(checkpoint_filename):
    """Return epoch and iteration number from filename '{filename}_{epoch}_{iteration}'."""
    checkpoint_filename = checkpoint_filename.split(".pt")[0]
    (epoch, iteration) = [int(string) for string in checkpoint_filename.split("_")[-2:]]
    return epoch, iteration


def setup_model(cfg: DictConfig):
    """Set up MENT-Flow model architecture from config."""   
    build_gen_kws = cfg["gen"]
    gen = mf.gen.build_gen(**build_gen_kws)
    
    model = mf.MENTFlow(
        gen=gen,
        prior=None, 
        entropy_estimator=None,
        transforms=None,
        diagnostics=None,
        measurements=None, 
    )
    return model


def load_model(cfg: dict, checkpoint_path: str):
    """Load MENT-Flow model architecture (from cfg) and parameters (from checkpoint_path)."""
    model = setup_model(cfg)
    model.load(checkpoint_path)
    return model


def load_run(folder):
    """Load data from training run."""
    cfg     = load_pickle_safe(os.path.join(folder, "cfg.pkl"))
    history = load_pickle_safe(os.path.join(folder, "history.pkl"))

    model = setup_model(cfg)
    model.eval()
    
    subdir = os.path.join(folder, "checkpoints")
    checkpoint_paths = [os.path.join(subdir, f) for f in sorted(os.listdir(subdir))]
    checkpoints = []
    for checkpoint_path in checkpoint_paths:
        (step, iteration) = get_epoch_and_iteration_number(checkpoint_path)
        checkpoint = {
            "step": step,
            "iteration": iteration,
            "path": checkpoint_path,
        }
        checkpoints.append(checkpoint)

    output = {
        "model": model,
        "checkpoints": checkpoints,
        "cfg": cfg,
        "history": history,
    }
    return output

