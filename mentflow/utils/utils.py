import itertools
import math
import pickle

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


def unravel(iterable):
    return itertools.chain.from_iterable(iterable)
    

def get_num_parameters(model):
    num_parameters = 0
    for parameter in model.parameters():
        num_parameters += torch.numel(parameter)
    return num_parameters


def grab(x):
    return x.detach().cpu().numpy()


def centers_from_edges(edges):
    return 0.5 * (edges[:-1] + edges[1:])


def load_pickle(path):
    with open(path, "rb") as file:
        return pickle.load(file)


def save_pickle(object, path):
    with open(path, "wb") as file:
        return pickle.dump(object, file)
