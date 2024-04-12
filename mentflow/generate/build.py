import inspect
import functools

import torch
import zuko

from .base import GenerativeModel
from .nn import NNTransform
from .nn import NNGenerator
from .flows import WrappedZukoFlow


def build_flow(
    name: str,
    input_features: int,
    output_features: int,
    hidden_layers: int,
    hidden_units: int,
    transforms: int,
    device=None,
    **kws    
) -> WrappedZukoFlow:
    """Build normalizing flow (mentflow.gen.WrappedZukoFlow)."""
    constructors = {
        "bpf": zuko.flows.BPF,
        "ffjord": zuko.flows.CNF,
        "gf": zuko.flows.GF,
        "maf": zuko.flows.MAF,
        "nag": zuko.flows.NAF,
        "nsf": zuko.flows.NSF,
        "sospf": zuko.flows.SOSPF,
        "unaf": zuko.flows.UNAF,
    }
    constructor = constructors[name]

    kws["features"] = output_features
    kws["hidden_features"] = hidden_layers * [hidden_units]
    kws["transforms"] = transforms

    flow = constructor(**kws)
    
    if name in ["maf", "nag", "nsf", "unaf"]:
        flow = zuko.flows.Flow(flow.transform.inv, flow.base)

    flow = flow.to(device)
    return WrappedZukoFlow(flow)


def build_nn(
    input_features: int,
    output_features: int,
    hidden_layers: int,
    hidden_units: int,
    dropout: int,
    activation: str,
    device=None,
) -> NNGenerator:
    """Build neural network generator (mentflow.gen.NN)."""
    loc = torch.zeros(input_features)
    loc = loc.type(torch.float32)
    cov = torch.eye(input_features)
    cov = cov.type(torch.float32)
    if device is not None:
        loc = loc.to(device)
        cov = cov.to(device)

    base = torch.distributions.MultivariateNormal(loc, cov)
    
    transform = NNTransform(
        input_features=input_features,
        output_features=output_features,
        hidden_layers=hidden_layers,
        hidden_units=hidden_units,
        dropout=dropout,
        activation=activation,
    )
    return NNGenerator(base, transform)


def build_generator(name: str, device: torch.device = None, **kws) -> GenerativeModel:
    """Build generative model.

    Parameters
    ----------
    name: str
        - Invertible neural networks (normalizing flows) (see zuko docs):
            - "bpf": zuko.flows.BPF,
            - "ffjord": zuko.flows.FFJORD
            - "gf": zuko.flows.GF
            - "gmm": zuko.flows.GMM
            - "maf": zuko.flows.MAF
            - "nag": zuko.flows.NAG
            - "nsf": zuko.flows.NSF
            - "sospf": zuko.flows.SOSPF
            - "unaf": zuko.flows.UNAF
        - Non-invertible neural networks:
            - "nn": mentflow.gen.NNGen
    device: torch.device
        Torch device.
    **kws
        Key word arguments are passed to the generative model constructor.

    Returns
    -------
    mentflow.gen.GenerativeModel
        Trainable generative model.
    """
    if name == "nn":
        return build_nn(device=device, **kws)
    elif name in [
        "bpf",
        "ffjord",
        "gf",
        "gmm",
        "maf",
        "nag",
        "nsf",
        "sospf",
        "unaf",
    ]:        
        return build_flow(name=name, device=device, **kws)
    else:
        raise ValueError(f"Invalid generative model name '{name}'")




