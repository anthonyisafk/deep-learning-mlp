import numpy as np


class Neuron:
    y:np.float32      # output
    f:callable        # activation function
    df:callable       # activation function derivative
    eta:np.float32    # learning rate
    _w:np.ndarray     # weights
    u:np.ndarray      # f(x)
    _d:np.float32     # target
    t:str             # [input, output, or hidden]
    n_in:int          # len(x)
    id:int            # id in the layer
    lid:int           # id of the layer
    _e:np.float32     # error value
    _delta:np.float32 # back propagation error signal


    def __init__(self, id, lid, w, t, eta, f, df):
        nw = len(w)
        self.u = 0
        self.y = 0
        self._delta = 0
        self.id = id
        self.lid = lid
        self._w = w
        self.t = t
        self.f = f
        self.df = df
        self.eta = eta
        self.n_in = nw - 1 if nw > 0 else 0


    def __str__(self):
        fmt = "node[L{},{},'{}'] : {}"
        return fmt.format(self.lid, self.id, self.t, self.w)


    def get_u(self, x:np.ndarray):
        return np.dot(x, self.w)


    def get_y(self):
        return self.f(self.u)


    def get_error(self):
        return self.d - self.y;

    # Properties that need to be turned `private`, to create setters. #

    @property
    def w(self):
        return self._w
    @w.setter
    def w(self, w):
        self._w = w


    @property
    def d(self):
        return self._d
    @d.setter
    def d(self, d):
        self._d = d


    @property
    def e(self):
        return self._e
    @e.setter
    def e(self, e):
        self._e = e


    @property
    def delta(self):
        return self._delta
    @delta.setter
    def delta(self, delta):
        self._delta = delta