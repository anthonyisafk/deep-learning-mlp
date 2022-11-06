from enum import Enum
import numpy as np


class activation(Enum):
    step_non_neg = 0
    step_neg = 1
    relu = 2
    tanh = 3
    logistic = 4


def nn_step(x):
    return 0 if x <= 0 else 1
def dnn_step(x):
    return 0

def n_step(x):
    return -1 if x <= 0 else 1
def dn_step(x):
    return 0

def relu(x):
    return 0 if x < 0 else x
def drelu(x):
    return 0 if x < 0 else 1

def tanh(x):
    return 1.5 * np.tanh(x)
def dtanh(x): # dtanh(x)/dx = (sech(x))^2
    return (0.67 / np.cosh(x)) ** 2

def logistic(x):
    return 1.5 / (1 + np.exp(-1 * x))
def dlogistic(x):
    return (1.5 * np.exp(-1 * x)) / ((1 + np.exp(-1 * x)) ** 2)


fs = [nn_step, n_step, relu, tanh, logistic]
dfs = [dnn_step, dn_step, drelu, dtanh, dlogistic]
def get_activation_function(strf:activation):
    return fs[strf.value], dfs[strf.value]

