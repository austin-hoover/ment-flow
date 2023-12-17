import numpy as np
import torch


class RotateXY(torch.nn.Module):
    def __init__(self, angle=(0.25 * np.pi)):
        super().__init__()
        self.R = torch.eye(4)
        self.R[0, 0] = +np.cos(angle)
        self.R[0, 2] = +np.sin(angle)
        self.R[2, 0] = -np.sin(angle)
        self.R[2, 2] = +np.cos(angle)
        
    def forward(self, x):
        return torch.matmul(x, self.R.T)

    def to(self, device):
        self.R = self.R.to(device)
        return self