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
    node.u = node.get_u(x)
    y = node.get_y()
    return node.u, y


def set_targets(out_layer, nout, d):
    for n in range(nout):
        out_layer.nodes[n].d = d[n]


def update_output_weights(last_hidden, out_layer, nout, eta, alpha):
    for i in range(nout):
        node_i = out_layer.nodes[i]
        wprev = node_i.wprev[0]
        node_i.wprev[0] = node_i.w[0]
        node_i.w[0] -= eta * node_i.delta + alpha * wprev
        for j in range(node_i.n_in):
            wprev = node_i.wprev[j + 1]
            node_i.wprev[j + 1] = node_i.w[j + 1]
            node_i.w[j + 1] += eta * node_i.delta * last_hidden.nodes[j].y + alpha * wprev


def calculate_delta(dfu, next_layer, i):
    delta = 0.0
    for j in range(next_layer.n):
        delta += next_layer.nodes[j].delta * next_layer.nodes[j].w[i + 1]
    return dfu * delta