from classes.network import *


def predict_node(node, x):
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


def calculate_delta(dfu, next_layer, next_layer_deltas, i):
    delta = 0.0
    for j in range(next_layer.n):
        delta += next_layer_deltas[j] * next_layer.nodes[j].w[i + 1]
    return dfu * delta
