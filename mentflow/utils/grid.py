import numpy as np
import torch


def coords_from_edges(edges: torch.Tensor) -> torch.Tensor:
    return 0.5 * (edges[:-1] + edges[1:])


def get_grid_points(*coords: torch.Tensor) -> torch.Tensor:
    return torch.vstack([C.ravel() for C in torch.meshgrid(*coords, indexing="ij")]).T
