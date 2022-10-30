from neuron import *


class Layer:
    def __init__(self, lid, n, n_prev, type, beta, theta, f):
        check_for_type(type)
        scal = np.sqrt(2 / n_prev) # factor by which the random weights will be multiplied.
        self.n = n
        self.f = f
        self.lid = lid
        self.type = type
        self.beta = beta
        self.nodes = np.ndarray(dtype=Neuron, shape=(n))
        for i in range(n):
            w = np.random.rand(n_prev + 1) * scal
            w[0] = -1
            self.nodes[i] = Neuron(i, lid, w, type, beta, theta, f) # `He` weight initialization.
            print(self.nodes[i])


    f:callable       # activation function
    beta:np.float32  # learning rate
    lid:int          # layer id
    n:int            # number of neurons
    type:str         # [input, output, or hidden]
    nodes:np.ndarray # list of neurons in layer
    e:np.float32     # sum of errors


def check_for_type(type):
    if type != 'i' and type != 'o' and type != 'h':
        error = f"No type '{type}' allowed. : {'i', 'o', 'h'}"
        raise Exception(error)