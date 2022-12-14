from classes.neuron import *
from utils.layer import check_for_type, initialize_node


class Layer:
    eta:np.float32   # learning rate
    lid:int          # layer id
    n:int            # number of neurons
    t:str            # [input, output, or hidden]
    nodes:np.ndarray # list of neurons in layer
    n_prev:int       # nodes in previous layer
    f:callable       # activation function of layer nodes
    df:callable      # df(x)/dx


    def __init__(self, lid, n, n_prev, t, eta, theta, f, df):
        check_for_type(t)
        # scal = np.sqrt(2 / n_prev) if lid != 0 else 1
        self.n = n
        self.lid = lid
        self.t = t
        self.eta = eta
        self.n_prev = n_prev
        self.f = f
        self.df = df
        self.nodes = np.ndarray(dtype=Neuron, shape=(n))
        for i in range(n):
            self.nodes[i] = initialize_node(i, lid, t, eta, theta, n_prev, f, df)
            # print(self.nodes[i])
