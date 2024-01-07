import itertools
import math
import pickle

import numpy as np
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