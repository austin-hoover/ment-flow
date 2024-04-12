from typing import List

import torch

from mentflow.types_ import Distribution


class GenerativeModel(Distribution, torch.nn.Module):
    """Base class for generative models."""
    def forward(self, z: torch.Tensor, **kws) -> torch.Tensor:
        raise NotImplementedError

    def inverse(self, x: torch.Tensor, **kws) -> torch.Tensor:
        raise NotImplementedError

    def forward_steps(self, z: torch.Tensor) -> List[torch.Tensor]:
        raise NotImplementedError
    
    def inverse_steps(self, x: torch.Tensor) -> List[torch.Tensor]:
        raise NotImplementedError

    def sample_base(self, size: int) -> torch.Tensor:
        raise NotImplementedError

    def dim(self) -> int:
        raise NotImplementedError
