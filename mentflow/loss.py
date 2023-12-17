from typing import Iterable
import torch


def mae(pred: torch.Tensor, targ: torch.Tensor) -> torch.Tensor:
    """Mean absolute error."""
    return torch.mean(torch.abs(pred - targ))


def mse(pred: torch.Tensor, targ: torch.Tensor) -> torch.Tensor:
    """Mean squared error."""
    return torch.mean(torch.square(pred - targ))


def kld(pred: torch.Tensor, targ: torch.Tensor, pad=1.00e-12) -> torch.Tensor:
    """Pointwise KL divergence."""
    log_pred = torch.log(pred + pad)
    return torch.nn.functional.kl_div(log_pred, targ, reduction="batchmean")


def get_loss_function(name):
    if name == "mae":
        return mae
    elif name == "mse":
        return mse
    elif name == "kld":
        return kld
    else:
        raise ValueError(f"Invalid loss function name {name}")
    