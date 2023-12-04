"""Tools for loading models."""
import os
import pickle
import sys

import torch
import torch.nn as nn
import zuko

import mentflow as mf


def get_epoch_and_iteration_number(checkpoint_filename):
    """Return epoch and iteration number from filename '{filename}_{epoch}_{iteration}'."""
    checkpoint_filename = checkpoint_filename.split(".pt")[0]
    (epoch, iteration) = [int(string) for string in checkpoint_filename.split("_")[-2:]]
    return epoch, iteration


def make_flow(
    features=2,
    transforms=5,
    spline_bins=20,
    hidden_units=64,
    hidden_layers=3,
    randperm=True,
    base=None,
):
    """Return normalizing flow model."""
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


def setup_model(cfg):
    """Setup MENT-Flow model architecture."""
    d = cfg["flow"]["features"]
    flow = make_flow(
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
        diagnostics=None
    )
    return model


def load_model(cfg, path):
    """Load MENT-Flow model architecture and parameters."""
    model = setup_model(cfg)
    model.load(path)
    return model


def load_run(folder):
    """Load all data from run."""
    args = mf.utils.load_pickle(os.path.join(folder, "args.pkl"))
    cfg = mf.utils.load_pickle(os.path.join(folder, "cfg.pkl"))
    dist = mf.utils.load_pickle(os.path.join(folder, "dist.pkl"))
    history = mf.utils.load_pickle(os.path.join(folder, "history.pkl"))

    model = setup_model(cfg)
    model.eval()
    
    checkpoints = []
    subdir = os.path.join(folder, "checkpoints")
    checkpoint_paths = [os.path.join(subdir, f) for f in sorted(os.listdir(subdir))]
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
        "args": args,
        "dist": dist,
        "history": history,
    }
    return output



## MENT-NN setup

def setup_model_nn(cfg):
    """Setup MENT-NN model architecture."""
    flow = mf.models.NNGenerator(
        input_features=cfg["flow"]["input_features"],
        output_features=cfg["flow"]["output_features"],
        hidden_layers=cfg["flow"]["hidden_layers"],
        hidden_units=cfg["flow"]["hidden_units"],
        dropout=cfg["flow"]["dropout"],
        activation=cfg["flow"]["activation"],
    )
    model = mf.MENTNN(
        d=cfg["flow"]["output_features"],
        flow=flow,
        base=cfg["base"],
        entropy_estimator=cfg["entropy_estimator"],
        target=None, 
        lattices=None,
        measurements=None, 
        diagnostics=None
    )
    return model


def load_model_nn(cfg, path):
    """Load MENT-NN model architecture and parameters."""
    model = setup_model_nn(cfg)
    model.load(path)
    return model


def load_run_nn(folder):
    """Load all data from run (MENT-NN)."""
    args = mf.utils.load_pickle(os.path.join(folder, "args.pkl"))
    cfg = mf.utils.load_pickle(os.path.join(folder, "cfg.pkl"))
    dist = mf.utils.load_pickle(os.path.join(folder, "dist.pkl"))
    history = mf.utils.load_pickle(os.path.join(folder, "history.pkl"))

    model = setup_model_nn(cfg)
    model.eval()
    
    checkpoints = []
    subdir = os.path.join(folder, "checkpoints")
    checkpoint_paths = [os.path.join(subdir, f) for f in sorted(os.listdir(subdir))]
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
        "args": args,
        "dist": dist,
        "history": history,
    }
    return output
