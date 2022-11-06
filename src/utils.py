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

def training_predict_node(node, x):
    u = node.get_u(x)
    y = node.get_y()
    return u, y


def set_targets(out_layer, nout, d):
    for n in range(nout):
        out_layer.nodes[n].d = d[n]


def update_output_weights(last_hidden, out_layer, nout, eta):
    for i in range(nout):
        node_i = out_layer.nodes[i]
        node_i.w[0] -= eta * node_i.delta
        for j in range(node_i.n_in):
            node_i.w[j + 1] += eta * node_i.delta * last_hidden.nodes[j].y


def calculate_delta(dfu, next_layer, i):
    delta = 0.0
    for j in range(next_layer.n):
        delta += dfu * next_layer.nodes[j].delta * next_layer.nodes[j].w[i]
    return delta