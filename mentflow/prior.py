import torch


class GaussianPrior:
    def __init__(self, ndim: int = 2, scale: float = 1.0, device=None) -> None:
        self.ndim = ndim
        self.scale = scale
        self.device = device
        self._initialize()

    def _initialize(self):
        loc = torch.zeros(self.ndim)
        loc = loc.type(torch.float32).to(self.device)
        
        cov = torch.eye(self.ndim) * (self.scale ** 2)
        cov = cov.type(torch.float32).to(self.device)

        self._dist = torch.distributions.MultivariateNormal(loc, cov)

    def to(self, device):
        self.device = device
        self._initialize()
        return self

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        return self._dist.log_prob(x)


class UniformPrior:
    def __init__(self, ndim: int = 2, scale: float = 100.0, device: torch.device = None) -> None:
        self.scale = scale
        self.ndim = ndim
        self.volume = scale ** self.ndim
        self.device = device

    def to(self, device):
        self.device = device
        return self

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        _log_prob = np.log(1.0 / self.volume)
        _log_prob = torch.ones(x.shape[0]) * _log_prob
        _log_prob = _log_prob.type(torch.float32)
        _log_prob = _log_prob.to(self.device)
        return _log_prob