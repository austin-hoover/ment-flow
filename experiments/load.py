"""Tools for loading models."""
import os
import pickle
import sys
import typing
from typing import Any
from typing import Callable
from typing import Dict

import torch
import torch.nn as nn
import zuko
from omegaconf import DictConfig
from omegaconf import OmegaConf

import mentflow as mf
from .setup import setup_mentflow_model


def load_pickle_safe(path):
    """Load pickled file; return None if file does not exist."""
    if os.path.exists(path):
        return mf.utils.load_pickle(path)
    return None


def epoch_and_iteration_number(checkpoint_filename):
    """Return epoch and iteration number from filename '{filename}_{epoch}_{iteration}'."""
    checkpoint_filename = checkpoint_filename.split(".pt")[0]
    (epoch, iteration) = [int(string) for string in checkpoint_filename.split("_")[-2:]]
    return epoch, iteration


def setup_ment_model(cfg: DictConfig) -> mf.MENTFlow:
    """Setup empty MENT model."""
    model = mf.alg.MENT(
        d=cfg.d,
        transforms=[],
        diagnostics=[],
        measurements=[],
    )
    return model


def load_mentflow_model(cfg: dict, checkpoint_path: str, device=None) -> mf.MENTFlow:
    """Load MENT-Flow model architecture (from cfg) and parameters (from checkpoint_path)."""
    model = setup_mentflow_model(cfg, transforms=None, diagnostics=None, measurements=None)
    model.load(checkpoint_path, device)
    return model


def load_ment_model(cfg: dict, checkpoint_path: str, device=None) -> mf.alg.MENT:
    """Load MENT model (from cfg) and parameters (from checkpoint_path)."""
    model = setup_ment_model(cfg)
    model.load(checkpoint_path, device)
    return model


def load_mentflow_run(folder: str, device=None) -> Dict:
    """Load data from MENT-Flow training run."""
    return load_run(folder, device, load_model=load_mentflow_model)
    

def load_ment_run(folder: str, device=None) -> Dict:
    """Load data from MENT training run."""
    return load_run(folder, device, load_model=load_ment_model)


def load_run(folder: str, device=None, load_model=None) -> Dict:
    cfg     = load_pickle_safe(os.path.join(folder, ".hydra/config.pickle"))
    history = load_pickle_safe(os.path.join(folder, "history.pkl"))
        
    checkpoints_folder = os.path.join(folder, "checkpoints")
    checkpoint_paths = os.listdir(checkpoints_folder)
    checkpoint_paths = [os.path.join(checkpoints_folder, f) for f in checkpoint_paths]
    checkpoint_paths = sorted(checkpoint_paths)
    checkpoints = []
    for checkpoint_path in checkpoint_paths:
        (epoch, iteration) = epoch_and_iteration_number(checkpoint_path)
        checkpoint = {
            "epoch": epoch,
            "iteration": iteration,
            "path": checkpoint_path,
        }
        checkpoints.append(checkpoint)

    model = load_model(cfg, checkpoint_paths[-1], device=device)
    
    output = {
        "model": model,
        "checkpoints": checkpoints,
        "config": cfg,
        "history": history,
    }
    return output


