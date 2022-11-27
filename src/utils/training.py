from classes.network import *


def predict_node(node, x):
    node.u = node.get_u(x)
    y = node.get_y()
    return node.u, y


def set_targets(out_layer, nout, d):
    for n in range(nout):
        out_layer.nodes[n].d = d[n]


'''
Iƒ a value is None, it means we don't want the training to stop earlier
because of it. Since we check for:
 - mse <= minJ,
 - accuracy >= min_acc,
we set:
 - minJ = 0
 - min_acc = 1.0
 - leave as is, otherwise.
'''
def get_min_training_values(minJ, min_acc):
    if minJ is None:
        minJ = 0.0
    if min_acc is None:
        min_acc = 1.0
    return minJ, min_acc


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


def check_for_early_stopping(e, acc, minJ, min_acc, iter):
    if e <= minJ:
        print(f"  >> Stopping training in step {iter}. MSE reached minJ = {minJ}, or lower.\n")
        return True
    if acc >= min_acc:
        print(f"  >> Stopping training in step {iter}. Accuracy reached reached {min_acc}, or higher.\n")
        return True
    return False
