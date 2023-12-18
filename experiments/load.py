"""Tools for loading models."""
import os
import pickle
import sys
import typing
from typing import Callable

import torch
import torch.nn as nn
import zuko

import mentflow as mf
import mentflow.wrappers


def load_pickle(path):
    """Load pickled file; return None if file does not exist."""
    if os.path.exists(path):
        return mf.utils.load_pickle(path)
    return None


def get_epoch_and_iteration_number(checkpoint_filename):
    """Return epoch and iteration number from filename '{filename}_{epoch}_{iteration}'."""
    checkpoint_filename = checkpoint_filename.split(".pt")[0]
    (epoch, iteration) = [int(string) for string in checkpoint_filename.split("_")[-2:]]
    return epoch, iteration


def make_generator_zuko(cfg):
    """Construct WrappedZukoFlow from config dict."""

    # Accept "features" or "output_features".
    features = None
    if "features" in cfg["generator"]:
        features = cfg["generator"]["features"]
    elif "output_features" in cfg["generator"]:
        features = cfg["generator"]["output_features"]
    else:
        raise ValueError("Need key 'features' or 'output_features'")

    # Create Neural Spline Flow (NSF) model.
    flow = zuko.flows.NSF(
        features=features,
        transforms=cfg["generator"]["transforms"],
        bins=cfg["generator"]["spline_bins"],
        hidden_features=(cfg["generator"]["hidden_layers"] * [cfg["generator"]["hidden_units"]]),
        randperm=cfg["generator"]["randperm"],
    )
    flow = zuko.flows.Flow(flow.transform.inv, flow.base)
    flow = mentflow.wrappers.WrappedZukoFlow(flow)
    return flow


def make_generator_nn(cfg):
    """Construct NNGenerator from config dict."""
    transformer = mf.models.NNTransformer(
        input_features=cfg["generator"]["input_features"],
        output_features=cfg["generator"]["output_features"],
        hidden_layers=cfg["generator"]["hidden_layers"],
        hidden_units=cfg["generator"]["hidden_units"],
        dropout=cfg["generator"]["dropout"],
        activation=cfg["generator"]["activation"],
    )
    base = torch.distributions.Normal(
        torch.zeros(cfg["generator"]["input_features"]),
        torch.ones(cfg["generator"]["input_features"])
    )
    return mf.models.NNGenerator(base, transformer)


def make_generator(cfg, generator_type="zuko"):
    """Constructe generative model from config dict."""
    _make_generator_options = {
        "zuko": make_generator_zuko,
        "nn": make_generator_nn,
    }
    _make_generator = _make_generator_options[generator_type]
    return _make_generator(cfg)
    

def setup_model(cfg: dict, generator_type="zuko"):
    """Set up MENT-Flow model architecture from config."""
    model = mf.MENTFlow(
        generator=make_generator(cfg, generator_type),
        prior=None, 
        entropy_estimator=None,
        transforms=None,
        diagnostics=None,
        measurements=None, 
    )
    return model


def load_model(cfg: dict, checkpoint_path: str, generator_type="zuko"):
    """Load MENT-Flow model architecture (from config) and parameters (from checkpoint)."""
    model = setup_model(cfg, generator_type)
    model.load(checkpoint_path)
    return model


def load_run(folder, generator_type="zuko"):  
    """Load all data from run."""
    args    = load_pickle(os.path.join(folder, "args.pkl"))
    cfg     = load_pickle(os.path.join(folder, "cfg.pkl"))
    dist    = load_pickle(os.path.join(folder, "dist.pkl"))
    history = load_pickle(os.path.join(folder, "history.pkl"))

    model = setup_model(cfg, generator_type)
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

    return {
        "model": model,
        "checkpoints": checkpoints,
        "args": args,
        "cfg": cfg,
        "dist": dist,
        "history": history,
    }
