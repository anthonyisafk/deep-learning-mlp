from neuron import *
from layer import *
from activation import *
from utils import *


class Network:
    nlayers:int                # number of layers
    layers:np.ndarray          # list of layers
    nodes_per_layer:np.ndarray # neurons per layer
    f:callable                 # activation function
    df:callable                # activation function derivative
    eta:np.float32             # learning rate
    alpha:np.float32           # momentum constant
    e:np.float32               # total error
    acc:np.float32             # accuracy percentage


    def __init__(self, nlayers, nodes, eta, theta, alpha, f:activation):
        check_for_layers(nlayers)
        self.nlayers = nlayers
        self.f, self.df = get_activation_function(f)
        self.nodes_per_layer = nodes
        self.eta = eta
        self.alpha = alpha
        self.e = 0
        self.layers = np.ndarray(dtype=Layer, shape=(nlayers))
        t = 'i'
        ident, dident = get_activation_function(activation.identity)
        self.layers[0] = Layer(0, nodes[0], 1, t, eta, theta, ident, dident)
        for i in range(1, nlayers):
            t = 'h' if i < nlayers - 1 else 'o'
            n_prev = nodes[i - 1]
            self.layers[i] = Layer(i, nodes[i], n_prev, t, eta, theta, self.f, self.df)


    def train(self, x, d, batch_size, epochs, minJ):
        size, last_idx, out_layer, nout = self.get_input_output_info(x)
        for iter in range(epochs):
            perm = np.random.permutation(size)
            # print(perm)
            x, d = x[perm], d[perm]
            curr_idx = 0
            self.e = 0
            for b in range(0, size, batch_size):
                b_size = min(batch_size, size - b)
                for p in range(b_size):
                    curr_x = x[curr_idx]
                    curr_d = d[curr_idx]
                    set_targets(out_layer, nout, curr_d)
                    y = self.training_predict(curr_x)
                    self.get_output_errors(nout, out_layer)
                    curr_idx += 1
                    last_hidden = self.layers[last_idx - 1]
                    update_output_weights(last_hidden, out_layer, nout, self.eta, self.alpha)
                    self.update_hidden_weights(last_hidden, out_layer, last_idx)
            self.e /= (size * nout)

            print(f"  ** epoch {iter} : e = {self.e}")
            # if self.e <= minJ:
            #     print(f"  >> Stopping training in step {iter}. MSE reached minJ = {minJ}, or lower.\n")
            #     break

    def training_predict(self, x):
        y = x
        for i in range(len(x)):
            self.layers[0].nodes[i].y = x[i]
        for i in range(1, self.nlayers):
            # print(y)
            y = self.training_predict_layer(y, i)
        return y


    def training_predict_layer(self, x, i):
        n_i = self.nodes_per_layer[i]
        layer_i = self.layers[i]
        y = np.ndarray(shape=(n_i))
        x = np.concatenate(([-1], x), axis=None)
        for j in range(n_i): # rest of the layers are fully connected.
            node_j = layer_i.nodes[j]
            node_j.u, node_j.y = training_predict_node(node_j, x)
            y[j] = node_j.y
        return y


    def get_input_output_info(self, x):
        size = len(x)
        last_idx = self.nlayers - 1
        out_layer = self.layers[last_idx]
        nout = self.nodes_per_layer[last_idx]
        return size, last_idx, out_layer, nout


    def get_output_errors(self, nout, out_layer):
        for n in range(nout):
            node = out_layer.nodes[n]
            node.e = node.get_error()
            node.delta = node.e * self.df(node.u)
            self.e += node.e ** 2


    def update_hidden_weights(self, last_hidden, out_layer, last_idx):
        curr_layer = last_hidden
        next_layer = out_layer
        for l in range(last_idx - 1, 0, -1):
            lsize = curr_layer.n
            previous_layer = self.layers[l - 1]
            for i in range(lsize):
                node_i = curr_layer.nodes[i]
                node_i.delta = calculate_delta(self.df(node_i.u), next_layer, i)
                wprev = node_i.wprev[0]
                node_i.wprev[0] = node_i.w[0]
                node_i.w[0] -= self.eta * node_i.delta + self.alpha * wprev
                for j in range(curr_layer.n_prev):
                    wprev = node_i.wprev[j + 1]
                    node_i.wprev[j + 1] = node_i.w[j + 1]
                    node_i.w[j + 1] += self.eta * node_i.delta * previous_layer.nodes[j].y + self.alpha * wprev
            # if l != 0:
            next_layer = curr_layer
            curr_layer = previous_layer

