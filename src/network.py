from neuron import *
from layer import *
from activation import *


class Network:
    def __init__(self, nlayers, nodes, eta, theta, epochs, minj, f:activation):
        self.nlayers = nlayers
        self.f, self.df = get_activation_function(f)
        self.nodes_per_layer = nodes
        self.eta = eta
        self.theta = theta
        self.epochs = epochs
        self.minj = minj
        self.layers = np.ndarray(dtype=Layer, shape=(nlayers))
        t = 'i'
        for i in range(nlayers):
            n_prev = nodes[i - 1] if i > 0 else 1
            self.layers[i] = Layer(i, nodes[i], n_prev, t, eta, theta, self.f, self.df)
            t = 'h' if i < nlayers - 2 else 'o'


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
                node_i.u = node_i.get_u(x[i])
                y[j] = node_i.get_y()
                node_i.y = y[j]
        else:
            for j in range(n_i): # rest of the layers are fully connected.
                node_i = self.layers[i].nodes[j]
                node_i.u = node_i.get_u(x)
                y[j] = node_i.get_y()
                node_i.y = y[j]
        return y


    nlayers:int                # number of layers
    layers:np.ndarray          # list of layers
    nodes_per_layer:np.ndarray # neurons per layer
    f:callable                 # activation function
    df:callable                # activation function derivative
    eta:np.float32             # learning rate
    theta:np.float32           # bias
    epochs:int                 # number of epochs per training
    e:np.float32               # total error
    acc:np.float32             # accuracy percentage
    minj:np.float32            # threshold by which training is considered done