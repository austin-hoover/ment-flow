"""Tools for loading models."""
import os
import pickle
import sys

import torch
import torch.nn as nn
import zuko

import mentflow as mf


def make_flow_nsf(
    features=2,
    transforms=5,
    spline_bins=20,
    hidden_units=64,
    hidden_layers=3,
    randperm=True,
    base=None,
):
    flow = zuko.flows.NSF(
        features=features,
        transforms=transforms,
        bins=spline_bins,
        hidden_features=(hidden_layers * [hidden_units]),
        randperm=randperm,
    )
    if base:
        flow.base = base
    flow = zuko.flows.Flow(flow.transform.inv, flow.base)
    return flow


def setup_model_nsf(cfg):
    """Setup NSF MENT-Flow model architecture."""
    d = cfg["flow"]["features"]
    flow = make_flow_nsf(
        features=d,
        transforms=cfg["flow"]["transforms"],
        spline_bins=cfg["flow"]["spline_bins"],
        hidden_units=cfg["flow"]["hidden_units"],
        hidden_layers=cfg["flow"]["hidden_layers"],
        randperm=cfg["flow"]["randperm"],
    )
    model = mf.MENTFlow(
        d=d, 
        flow=flow,
        target=None, 
        lattices=None,
        measurements=None, 
        diagnostic=None
    )
    return model


def load_model_nsf(cfg, path):
    """Load NSF MENT-Flow model architecture and parameters."""
    model = setup_model(cfg)
    model.load(path)
    return model


def get_epoch_and_iteration_number(checkpoint_filename):
    """Return epoch and iteration number from filename '{filename}_{epoch}_{iteration}'."""
    checkpoint_filename = checkpoint_filename.split(".pt")[0]
    (epoch, iteration) = [int(string) for string in checkpoint_filename.split("_")[-2:]]
    return epoch, iteration


def load_info(path):
    """Load text file with command line arguments as dict."""
    info = {}
    file = open(path, "r")
    for line in file:
        line = line.rstrip()
        if line.startswith("Namespace"):
            line = line.split("Namespace(")[1].split(")")[0]
            strings = [string.strip() for string in line.split(",")]
            for string in strings:
                key, val = string.split("=")
                info[key] = val
    file.close()
    return info


def load_run(folder):
    """Load model, checkpoints, config, info."""
    
    # Load config dict.
    path = os.path.join(folder, "config.pkl")
    with open(path, "rb") as file:
        cfg = pickle.load(file)
    
    # Load command line arguments.
    path = os.path.join(folder, "log.txt")
    info = load_info(path)
    
    # Set up the model.
    model = setup_model(cfg)
    model.eval()
    
    # Load checkpoint paths.
    checkpoints = []
    subdir = os.path.join(folder, "checkpoints")
    checkpoint_paths = [os.path.join(subdir, f) for f in sorted(os.listdir(subdir))]
    for checkpoint_path in checkpoint_paths:
        (step, iteration) = get_step_and_iteration_number(checkpoint_path)
        checkpoint = {
            "step": step,
            "iteration": iteration,
            "path": checkpoint_path,
        }
        checkpoints.append(checkpoint)
        
    return (model, checkpoints, cfg, info)