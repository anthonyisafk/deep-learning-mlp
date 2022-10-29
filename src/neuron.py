import numpy as np

class Neuron:
    def __init__(self, id, lid, w, type, n_in, n_out, f):
        self._id = id
        self._lid = lid
        self._w = w
        self._type = type
        self._n_in = n_in
        self._n_out = n_out
        self.f = f
        self.y = np.ndarray
        self._x = np.ndarray
        self._w = np.ndarray


    def __str__(self):
        txt = "node[{}(L{})] ({})"
        return txt.format(self._id, self._lid, self._type)


    def get_y(self):
        self._u = np.dot(self._x, self._w)
        self.y = self.f(self._u)


    y:np.ndarray
    f:callable
    _x:np.ndarray
    _w:np.ndarray
    _u:np.ndarray
    _d:float
    _type:str
    _n_in:int
    _n_out:int
    _id:int
    _lid:int