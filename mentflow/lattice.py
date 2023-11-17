import torch


class Lattice(torch.nn.Module):
    """Represents an accelerator lattice."""
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        return NotImplementedError


class LinearLattice(Lattice):
    """Represents a linear lattice."""
    def __init__(self) -> None:
        super().__init__()
        self.set_matrix(torch.eye(6))

    def set_matrix(self, matrix: torch.Tensor) -> None:
        self.matrix = matrix
        self.matrix_inv = torch.linalg.inv(self.matrix)   
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(x, self.matrix.T)

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(x, self.matrix_inv.T)

    def to(self, device):
        self.matrix = self.matrix.to(device)
        self.matrix_inv = self.matrix_inv.to(device)
        return self
