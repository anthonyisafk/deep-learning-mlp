from enum import Enum
import numpy as np
from mpmath import sech


alpha = 1.7159
beta = 2 / 3

class activation(Enum):
    identity = 0
    relu = 1
    tanh = 2
    logistic = 3


def identity(x):
    return x
def didentity(x):
    return 0


def relu(x):
    return 0 if x < 0 else x
def drelu(x):
    return 0 if x < 0 else 1


def tanh(x):
    return alpha * np.tanh(beta * x)
def dtanh(x): # d(atanh(bx))/dx = ab(sech(bx))^2
    return alpha * beta * sech(beta * x) ** 2


def logistic(x):
    return 1.0 / (1 + np.exp(-1 * x))
def dlogistic(x):
    return (np.exp(-1 * x)) / ((1 + np.exp(-1 * x)) ** 2)


fs = [identity, relu, tanh, logistic]
dfs = [didentity, drelu, dtanh, dlogistic]
def get_activation_function(strf:activation):
    return fs[strf.value], dfs[strf.value]

