from typing import Iterable
import torch
import ot


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


class SlicedWassersteindDistance:
    def __init__(self, n_projections=50, p=2, device=None):
        self.n_projections = n_projections
        self.p = p
        self.device = device
        if self.device is None:
            self.device = torch.device("cpu")
    
    def __call__(self, x1, x2): 
        if x1.shape[1] != x2.shape[1]:
            raise ValueError(f"x1.shape[1]={x1.shape} != x2.shape[1]={x2.shape})")

        p = self.p
        n_projections = self.n_projections
        
        d = x1.shape[1]
        directions = torch.randn(d, n_projections, device=self.device)
        directions = directions / torch.sqrt(torch.sum(directions**2, 0, keepdims=True))

        x1_projections = torch.matmul(x1, directions)
        x2_projections = torch.matmul(x2, directions)
        projected_emd = ot.lp.wasserstein_1d(x1_projections, x2_projections, p=p)
        distance = (torch.sum(projected_emd) / n_projections) ** (1.0 / p)
        return distance



