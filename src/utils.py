import numpy as np
from neuron import *


# ---------- LAYER UTILITIES ---------- #

"""
Initialize a neuron using He random weight assignment.
"""
def initialize_node(i, lid, t, eta, theta, n_prev, scal, f, df):
    w = np.random.rand(n_prev + 1) * scal
    w[0] = theta
    return Neuron(i, lid, w, t, eta, f, df)


def check_for_type(t):
    if t != 'i' and t != 'o' and t != 'h':
        error = f"No type '{t}' allowed. : {'i', 'o', 'h'}"
        raise Exception(error)


# ---------- NETWORK TRAINING UTILITIES ---------- #

def training_predict_first_layer(node, x):
    u = node.get_u(x)
    y = node.get_y()
    return u, y


def training_predict_hidden_layer(node, x):
    u = node.get_u(x)
    y = node.get_y()
    return u, y
