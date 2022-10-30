import numpy as np


class Neuron:
    def __init__(self, id, lid, w, type, beta, theta, f):
        nw = len(w)
        self.id = id
        self.lid = lid
        self.w = w
        self.type = type
        self.f = f
        self.beta = beta
        self.n_in = nw - 1 if nw > 0 else 0
        self.x = np.ndarray(dtype=np.float16, shape=(nw))
        self.x[0] = theta


    def __str__(self):
        txt = "node[L{},{}] ({}) : {}"
        return txt.format(self.lid, self.id, self.type, self.w)


    def get_y(self):
        self.u = np.dot(self.x, self.w)
        self.y = self.f(self.u)


    y:np.float32    # output
    f:callable      # activation function
    beta:np.float32 # learning rate
    x:np.ndarray    # input
    w:np.ndarray    # weights
    u:np.ndarray    # f(x)
    d:float         # targets
    type:str        # [input, output, or hidden]
    n_in:int        # len(x)
    id:int          # id in the layer
    lid:int         # id of the layer