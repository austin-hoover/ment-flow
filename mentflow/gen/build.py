import inspect
import functools

import torch
import zuko

from .gen import GenModel
from .nn import NNGen
from .nn import NNTransformer
from .wrappers import WrappedZukoFlow


def build_flow(
    name: str,
    input_features: int,
    output_features: int,
    hidden_layers: int,
    hidden_units: int,
    transforms: int,
    invert=True,
    **kws    
) -> WrappedZukoFlow:
    """Build normalizing flow (zuko)."""
    constructors = {
        "ffjord": zuko.flows.CNF,
        "gf": zuko.flows.GF,
        "gmm": functools.partial(zuko.flows.GMM, components=16),
        "maf": zuko.flows.MAF,
        "nag": zuko.flows.NAF,
        "nsf": zuko.flows.NSF,
        "sospf": zuko.flows.SOSPF,
        "unaf": zuko.flows.UNAF,
    }
    constructor = constructors[name]

    kws["features"] = output_features
    kws["hidden_features"] = hidden_layers * [hidden_units]
    if "transforms" in inspect.signature(constructor).parameters:
        kws["transforms"] = transforms

    flow = constructor(**kws)
    if invert:
        flow = zuko.flows.Flow(flow.transform.inv, flow.base)
    return WrappedZukoFlow(flow)


def build_nn(
    input_features: int,
    output_features: int,
    hidden_layers: int,
    hidden_units: int,
    dropout: int,
    activation: str,
) -> NNGen:
    """Build neural network generator.

    Key word arguments passed to mentflow.gen.NNTransformer.
    """
    base = torch.distributions.Normal(
        torch.zeros(input_features),
        torch.ones(input_feaetures),
    )
    transformer = NNTransformer(
        input_features=input_features,
        output_features=output_features,
        hidden_layers=hidden_layers,
        hidden_units=hidden_units,
        dropout=dropout,
        activation=activation,
    )
    return NNGen(base, transformer)


def build_gen(name: str, **kws) -> GenModel:
    """Build generative model.

    Parameters
    ----------
    name: str
        - Invertible neural networks (see zuko docs):
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
    **kws
        Key word arguments are passed to the model constructor.

    Returns
    -------
    mentflow.gen.GenModel
        Trainable generative model.
    """
    build = None
    if name == "nn":
        return build_nn(**kws)
    else:
        return build_flow(name, **kws)
