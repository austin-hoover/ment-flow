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


def augmented_lagrangian(
    H: torch.Tensor,
    C: torch.Tensor,
    lagrange_multipliers: Iterable[float],
    penalty_parameter: float,
) -> torch.Tensor:
    """Augmented Lagrangian: L = H + lambda * C + mu * C^2."""
    return H + sum(
        l * c + penalty_parameter * c**2 for l, c in zip(lagrange_multipliers, C)
    )
