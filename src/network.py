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
        layers = np.ndarray(dtype=Layer, shape=(nlayers))
        type = 'i'
        for i in range(nlayers):
            n_prev = nodes[i - 1] if i > 0 else 1
            layers[i] = Layer(i, nodes[i], n_prev, type, beta, theta, f)
            type = 'h' if i < nlayers - 2 else 'o'


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