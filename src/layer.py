from neuron import *


class Layer:
    def __init__(self, lid, n, n_prev, t, eta, theta, f, df):
        check_for_type(t)
        scal = np.sqrt(2 / n_prev) # factor by which the random weights will be multiplied.
        self.n = n
        self.lid = lid
        self.t = t
        self.eta = eta
        self.nodes = np.ndarray(dtype=Neuron, shape=(n))
        for i in range(n):
            self.nodes[i] = initialize_node(i, lid, t, eta, theta, n_prev, scal, f, df)


    eta:np.float32   # learning rate
    lid:int          # layer id
    n:int            # number of neurons
    t:str            # [input, output, or hidden]
    nodes:np.ndarray # list of neurons in layer
    e:np.float32     # sum of errors


"""
Initialize a neuron using He random weight assignment.
"""
def initialize_node(i, lid, t, eta, theta, n_prev, scal, f, df):
    w = np.random.rand(n_prev + 1) * scal
    w[0] = -1
    return Neuron(i, lid, w, t, eta, theta, f, df)


def check_for_type(t):
    if t != 'i' and t != 'o' and t != 'h':
        error = f"No type '{t}' allowed. : {'i', 'o', 'h'}"
        raise Exception(error)