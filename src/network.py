from neuron import *
from layer import *


class Network:
    def __init__(self, nlayers, nodes, beta, theta, epochs, minj, f):
        self.nlayers = nlayers
        self.f = f
        self.nodes_per_layer = nodes
        self.beta = beta
        self.theta = theta
        self.epochs = epochs
        self.minj = minj
        self.layers = np.ndarray(dtype=Layer, shape=(nlayers))
        type = 'i'
        for i in range(nlayers):
            n_prev = nodes[i - 1] if i > 0 else 1
            self.layers[i] = Layer(i, nodes[i], n_prev, type, beta, theta, f)
            type = 'h' if i < nlayers - 2 else 'o'


    def predict(self, x):
        y = x
        for i in range(self.nlayers):
            print(y)
            y = self.predict_layer(y, i)
        return y


    def predict_layer(self, x, i):
        n_i = self.nodes_per_layer[i]
        y = np.ndarray(shape=(n_i))
        if i == 0: # input layer only `sees` one entry of the input.
            for j in range(n_i):
                node_i = self.layers[i].nodes[j]
                node_i.set_y(x[j])
                y[j] = node_i.y
        else:
            for j in range(n_i): # rest of the layers are fully connected.
                node_i = self.layers[i].nodes[j]
                node_i.set_y(x)
                y[j] = node_i.y
        return y


    nlayers:int                # number of layers
    layers:np.ndarray          # list of layers
    nodes_per_layer:np.ndarray # neurons per layer
    f:callable                 # activation function
    beta:np.float32            # learning rate
    theta:np.float32           # bias
    epochs:int                 # number of epochs per training
    e:np.float32               # total error
    acc:np.float32             # accuracy percentage
    minj:np.float32            # threshold by which training is considered done