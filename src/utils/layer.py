from classes.neuron import *


def check_for_layers(nlayers):
    if nlayers < 2:
        error = f"Number of layers given was less than 2 : {nlayers}. " + \
        "The script makes an MLP with at least 2 layers."
        raise Exception(error)


def initialize_node(i, lid, t, eta, theta, n_prev, f, df):
    if t != 'i':
        np.random.seed(i * lid)
        w = np.random.uniform(size=(n_prev + 1))
        w[0] = theta
        return Neuron(i, lid, w, t, eta, f, df)
    else:
        w = np.array([1])
        return Neuron(i, lid, w, t, eta, f, df)


def check_for_type(t):
    if t != 'i' and t != 'o' and t != 'h':
        error = f"No type '{t}' allowed. : {'i', 'o', 'h'}"
        raise Exception(error)