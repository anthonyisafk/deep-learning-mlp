from classes.neuron import *
from classes.layer import *
from activation import *
from utils.layer import *
from utils.training import *
import time


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
    sdeltas:np.ndarray         # table of deltas for a single sample.


    def __init__(self, nodes, eta, theta, alpha, f:activation):
        self.nlayers = len(nodes)
        check_for_layers(self.nlayers)
        self.f, self.df = get_activation_function(f)
        self.nodes_per_layer = nodes
        self.eta = eta
        self.alpha = alpha
        self.e = 0
        self.layers = np.ndarray(dtype=Layer, shape=(self.nlayers))
        t = 'i'
        ident, dident = get_activation_function(activation.identity)
        self.layers[0] = Layer(0, nodes[0], 1, t, eta, theta, ident, dident)
        for i in range(1, self.nlayers):
            t = 'h' if i < self.nlayers - 1 else 'o'
            n_prev = nodes[i - 1]
            self.layers[i] = Layer(i, nodes[i], n_prev, t, eta, theta, self.f, self.df)
        self.sdeltas = np.ndarray(dtype=np.ndarray, shape=(self.nlayers - 1))
        for i in range(self.nlayers - 1):
            self.sdeltas[i] = np.zeros(dtype=np.float32, shape=(nodes[i + 1]))


    def train(self, x, d, batch_size, epochs, minJ):
        fmt = "  * Epoch {} [{:.2} secs.]: e = {:.2}, accuracy = {:.2}"
        train_start = time.time()
        size, last_idx, out_layer, nout = self.get_input_output_info(x)
        for iter in range(epochs):
            epoch_start = time.time()
            perm = np.random.permutation(size)
            x, d = x[perm], d[perm]
            curr_idx = 0
            self.e = 0.0
            self.acc = 0.0
            for b in range(0, size, batch_size):
                b_size = min(batch_size, size - b)
                for p in range(b_size):
                    curr_x = x[curr_idx]
                    curr_d = d[curr_idx]
                    set_targets(out_layer, nout, curr_d)
                    y = self.predict(curr_x)
                    self.update_output_errors(nout, out_layer)
                    self.update_hidden_deltas(out_layer, last_idx)
                    if np.argmax(y) == np.argmax(curr_d):
                        self.acc += 1
                    curr_idx += 1
                last_hidden = self.layers[last_idx - 1]
                update_output_weights(last_hidden, out_layer, nout, self.eta, self.alpha)
                self.update_hidden_weights(last_hidden, last_idx)
                self.reset_errors_and_deltas(nout, out_layer, last_idx)
            self.e /= (size * nout)
            self.acc /= size
            epoch_end = time.time() - epoch_start
            print(fmt.format(iter, epoch_end, self.e, self.acc))
            # if self.e <= minJ:
            #     print(f"  >> Stopping training in step {iter}. MSE reached minJ = {minJ}, or lower.\n")
            #     break
        train_time = time.time() - train_start
        print(f"\n  >> Training took {train_time:.2} secs. for {epochs} epochs.")
        print(f"   -- Accuracy : {self.acc:.2}\n")


    def predict(self, x):
        y = x
        for i in range(len(x)):
            self.layers[0].nodes[i].y = x[i]
        for i in range(1, self.nlayers):
            # print(y)
            y = self.predict_layer(y, i)
        return y


    def predict_layer(self, x, i):
        n_i = self.nodes_per_layer[i]
        layer_i = self.layers[i]
        y = np.ndarray(shape=(n_i))
        x = np.concatenate(([-1], x), axis=None)
        for j in range(n_i): # rest of the layers are fully connected.
            node_j = layer_i.nodes[j]
            node_j.u, node_j.y = predict_node(node_j, x)
            y[j] = node_j.y
        return y


    def get_input_output_info(self, x):
        size = len(x)
        last_idx = self.nlayers - 1
        out_layer = self.layers[last_idx]
        nout = self.nodes_per_layer[last_idx]
        return size, last_idx, out_layer, nout


    def update_output_errors(self, nout, out_layer):
        for n in range(nout):
            node = out_layer.nodes[n]
            e = node.get_error()
            delta = e * self.df(node.u)
            node.e += e
            self.sdeltas[self.nlayers - 2][n] = delta
            node.delta += delta
            self.e += e ** 2


    def update_hidden_deltas(self, out_layer, last_idx):
        next_layer = out_layer
        for l in range(last_idx - 1, 0, -1):
            curr_layer = self.layers[l]
            for i in range(curr_layer.n):
                curr_layer.nodes[i].delta += calculate_delta(
                    self.df(curr_layer.nodes[i].u), next_layer, self.sdeltas[l], i
                )
            next_layer = curr_layer


    def update_hidden_weights(self, last_hidden, last_idx):
        curr_layer = last_hidden
        for l in range(last_idx - 1, 0, -1):
            lsize = curr_layer.n
            previous_layer = self.layers[l - 1]
            for i in range(lsize):
                node_i = curr_layer.nodes[i]
                wprev = node_i.wprev[0]
                node_i.wprev[0] = node_i.w[0]
                node_i.w[0] -= self.eta * node_i.delta + self.alpha * wprev
                for j in range(curr_layer.n_prev):
                    wprev = node_i.wprev[j + 1]
                    node_i.wprev[j + 1] = node_i.w[j + 1]
                    node_i.w[j + 1] += self.eta * node_i.delta * previous_layer.nodes[j].y + self.alpha * wprev
            # if l != 0:
            curr_layer = previous_layer


    def reset_errors_and_deltas(self, nout, out_layer, last_idx):
        for i in range(nout):
            out_layer.nodes[i].e = 0
            out_layer.nodes[i].delta = 0
        for l in range(last_idx - 1, 0, -1):
            layer = self.layers[l]
            for i in range(layer.n):
                layer.nodes[i].delta = 0

