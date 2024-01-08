from typing import List
from typing import Tuple

import torch
import zuko

from mentflow.gen import GenModel


class WrappedZukoFlow(GenModel):
    def __init__(self, flow: zuko.flows.Flow):
        super().__init__()
        self._flow = flow

    def sample(self, n: int) -> torch.Tensor:
        return self._flow().rsample((n,))

    def sample_base(self, n: int) -> torch.Tensor:
        return self._flow().base.sample((n,))

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        return self._flow().log_prob(x)

    def sample_and_log_prob(self, n: int) -> Tuple[torch.Tensor, torch.Tensor]:
        (x, log_prob) = self._flow().rsample_and_log_prob((n,))
        return (x, log_prob)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self._flow().transform.inv(z)

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        return self._flow().transform(x)

    def forward_steps(self, z: torch.Tensor) -> List[torch.Tensor]:
        flow = self._flow()
        x = z.clone()
        xt = [x]
        for transform in flow.transform.inv.transforms:
            x = transform(x)
            xt.append(x)
        return xt

    def inverse_steps(self, x: torch.Tensor) -> List[torch.Tensor]:
        flow = self._flow()
        z = x.clone()
        zt = [z]
        for transform in flow.transform.transforms:
            z = transform(z)
            zt.append(z)
        return zt

    def get_flow(self):
        return self._flow()

        
