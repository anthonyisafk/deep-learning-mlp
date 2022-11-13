"""
Neural Networks & Deep Learning
Aristotle University Thessaloniki - School of Informatics.
******************************
@brief: An *attempt* at making an ANN from scratch.
        Trained on the `Iris` dataset.
@author: Antoniou, Antonios - 9482
@email: aantonii@ece.auth.gr
2022 AUTh Electrical and Computer Engineering.
"""

import pandas as pd
from network import *
from utils.testing import *


train_fraction = 0.7
minJ = 0.05
alpha = 2e-6
theta = 0.0
eta = 0.025
epochs = 20
batch_size = 2
f = activation.tanh

species = {
    'Iris-setosa': 0,
    'Iris-versicolor': 1,
    'Iris-virginica': 2,
}

def main():
    df = pd.read_csv('iris/iris.csv', header=0)
    nrows = len(df.values[:, 0])
    ncols = len(df.values[0, :])
    x = df.values[:, 1:ncols - 1]
    y = np.zeros(shape=(nrows,3))
    for i in range(nrows):
        y[i, species[df.values[i, ncols-1]]] = 1.0

    samples, targets = split_into_classes(x, y)
    x_train, x_test, y_train, y_test = split_trainset_testset(samples, targets)

    nin = len(x[0])
    nout = len(y[0])
    nodes = [nin, 7, nout]
    mlp = Network(nodes, eta, theta, alpha, f)
    mlp.train(x_train, y_train, batch_size, epochs, minJ)

    test_rate = test_network(x_test, y_test, mlp, True)


def split_into_classes(x, y):
    samples = [[] for _ in range(3)]
    targets = [[] for _ in range(3)]
    for v in species.values():
        indices = np.asarray(y[:, v] == 1.0)
        samples[v] = x[indices]
        targets[v] = y[indices]
    return samples, targets


def split_trainset_testset(samples, targets):
    x_train = []
    x_test = []
    y_train = []
    y_test = []
    for v in species.values():
        nv = len(samples[v])
        ntrain = int(train_fraction * nv)
        if v == 0:
            x_train = samples[v][0:ntrain]
            x_test = samples[v][ntrain:]
            y_train = targets[v][0:ntrain]
            y_test = targets[v][ntrain:]
        else:
            x_train = np.concatenate((x_train, samples[v][0:ntrain]))
            x_test = np.concatenate((x_test, samples[v][ntrain:]))
            y_train = np.concatenate((y_train, targets[v][0:ntrain]))
            y_test = np.concatenate((y_test, targets[v][ntrain:]))
    return x_train, x_test, y_train, y_test


if __name__ == "__main__":
    main()