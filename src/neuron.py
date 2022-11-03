import numpy as np


class Neuron:
    def __init__(self, id, lid, w, type, eta, theta, f):
        nw = len(w)
        self.id = id
        self.lid = lid
        self.w = w
        self.type = type
        self.f = f
        self.eta = eta
        self.theta = theta
        self.n_in = nw - 1 if nw > 0 else 0


    def __str__(self):
        txt = "node[L{},{}] ({}) : {}"
        return txt.format(self.lid, self.id, self.type, self.w)


    def set_y(self, x):
        x = np.concatenate(([self.theta], x), axis=None)
        self.u = np.dot(x, self.w)
        self.y = self.f(self.u)


    y:np.float32     # output
    f:callable       # activation function
    theta:np.float32 # bias
    eta:np.float32   # learning rate
    w:np.ndarray     # weights
    u:np.ndarray     # f(x)
    d:np.float32     # target
    type:str         # [input, output, or hidden]
    n_in:int         # len(x)
    id:int           # id in the layer
    lid:int          # id of the layer