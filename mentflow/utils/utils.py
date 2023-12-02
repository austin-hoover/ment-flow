import numpy as np
import torch


def is_bool(x):
    return isinstance(x, bool)


def is_int(x):
    return isinstance(x, int)


def is_positive_int(x):
    return is_int(x) and x > 0


def is_nonnegative_int(x):
    return is_int(x) and x >= 0


def is_power_of_two(n):
    if is_positive_int(n):
        return not n & (n - 1)
    else:
        return False


def get_num_parameters(model):
    num_parameters = 0
    for parameter in model.parameters():
        num_parameters += torch.numel(parameter)
    return num_parameters


def tensor_to_numpy(x):
    return x.detach().cpu().numpy()


def rotation_matrix(angle):
    _cos = np.cos(angle)
    _sin = np.sin(angle)
    return np.array([[_cos, _sin], [-_sin, _cos]])


def centers_from_edges(edges):
    return 0.5 * (edges[:-1] + edges[1:])
