"""Tools for loading model."""
import torch
import torch.nn as nn
import zuko

import mentflow as mf


def make_flow_nsf(
    d=2,
    transforms=5,
    spline_bins=20,
    hidden_units=64,
    hidden_layers=3,
    randperm=True,
    base=None,
):
    flow = zuko.flows.NSF(
        features=d,
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
    d = cfg["flow"]["d"]
    flow = make_flow_nsf(
        d=d,
        transforms=cfg["flow"]["transforms"],
        spline_bins=cfg["flow"]["spline_bins"],
        hidden_units=cfg["flow"]["hidden_units"],
        hidden_layers=cfg["flow"]["hidden_layers"],
        randperm=cfg["flow"]["randperm"],
    )
    model = mf.MENTFlow(d=d, flow=flow, target=None, lattices=None, measurements=None, diagnostic=None)
    return model


def load_model(cfg, filename):
    model = setup_model(cfg)
    model.load(fileame)
    return model


def get_step_and_iteration_number(checkpoint_filename):
    checkpoint_filename = checkpoint_filename.split(".pt")[0]
    (step, iteration) = [int(string) for string in checkpoint_filename.split("_")[-2:]]
    return step, iteration