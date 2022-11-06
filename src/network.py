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
    e:np.float32               # total error
    acc:np.float32             # accuracy percentage


    def __init__(self, nlayers, nodes, eta, theta, f:activation):
        self.nlayers = nlayers
        self.f, self.df = get_activation_function(f)
        self.nodes_per_layer = nodes
        self.eta = eta
        self.e = 0
        self.layers = np.ndarray(dtype=Layer, shape=(nlayers))
        t = 'i'
        for i in range(nlayers):
            n_prev = nodes[i - 1] if i > 0 else 1
            self.layers[i] = Layer(i, nodes[i], n_prev, t, eta, theta, self.f, self.df)
            t = 'h' if i < nlayers - 2 else 'o'


    def train(self, x, d, batch_size, epochs, minJ):
        size, last_idx, out_layer, nout = self.get_input_output_info(x)
        for iter in range(epochs):
            curr_idx = 0
            self.e = 0
            for b in range(0, size, batch_size):
                b_size = min(batch_size, size - b * batch_size)
                for p in range(b_size):
                    curr_x = x[curr_idx]
                    curr_d = d[curr_idx]
                    set_targets(out_layer, nout, curr_d)
                    y = self.training_predict(curr_x)
                    self.get_output_errors(nout, out_layer)
                last_hidden = self.layers[last_idx - 1]
                update_output_weights(last_hidden, out_layer, nout, self.eta)
                self.update_hidden_weights(last_hidden, out_layer, last_idx)
                curr_idx += 1
            self.e /= size

            print(f"  ** iter {iter} : e = {self.e}")
            # if self.e <= minJ:
            #     print(f"  >> Stopping training in step {iter}. MSE reached minJ = {minJ}, or lower.\n")
            #     break

    def training_predict(self, x):
        y = x
        for i in range(self.nlayers):
            # print(y)
            y = self.training_predict_layer(y, i)
        return y


    def training_predict_layer(self, x, i):
        n_i = self.nodes_per_layer[i]
        layer_i = self.layers[i]
        y = np.ndarray(shape=(n_i))
        if i == 0: # input layer only `sees` one entry of the input.
            for j in range(n_i):
                node_j = layer_i.nodes[j]
                node_j.u, node_j.y = training_predict_node(node_j, [-1, x[j]])
                y[j] = node_j.y
        else:
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
        for l in range(last_idx - 1, -1, -1):
            lsize = curr_layer.n
            previous_layer = self.layers[l - 1]
            for i in range(lsize):
                node_i = curr_layer.nodes[i]
                node_i.delta = calculate_delta(self.df(node_i.u), next_layer, i)
                node_i.w[0] -= self.eta * node_i.delta
                for j in range(curr_layer.n_prev):
                    node_i.w[j + 1] += self.eta * node_i.delta * previous_layer.nodes[j].y
            if l != 0:
                next_layer = curr_layer
                curr_layer = previous_layer

