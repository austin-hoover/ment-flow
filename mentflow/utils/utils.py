import itertools
import math
import pickle

import numpy as np
import scipy.special
import torch


def unravel(iterable):
    return itertools.chain.from_iterable(iterable)
    

def grab(x):
    return x.detach().cpu().numpy()


def load_pickle(path):
    with open(path, "rb") as file:
        return pickle.load(file)


def save_pickle(object, path):
    with open(path, "wb") as file:
        return pickle.dump(object, file)


def exp_avg(values, momentum=0.99):
    values_smooth = np.zeros(len(values))
    y = values[0]
    for i, x in enumerate(values):
        y = momentum * y + (1.0 - momentum) * x
        values_smooth[i] = y
    return values_smooth


def sphere_surface_area(r=1.0, d=3):
    factor = 2.0 * np.pi ** (0.5 * d)
    factor = factor / scipy.special.gamma(0.5 * d)
    return factor * (r ** (d - 1))


def sphere_volume(r=1.0, d=3):
    factor = (np.pi ** (0.5 * d)) / scipy.special.gamma(1.0 + 0.5 * d)
    return factor * (r ** d)


def sphere_shell_volume(rmin=0.0, rmax=1.0, d=3):
    return sphere_volume(r=rmax, d=d) - sphere_volume(r=rmin, d=d)

