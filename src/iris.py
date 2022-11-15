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
from classes.network import *
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

    samples, targets = split_into_classes(x, y, 3, species)
    x_train, x_test, y_train, y_test = split_trainset_testset(samples, targets, train_fraction, species)

    nin = len(x[0])
    nout = len(y[0])
    nodes = [nin, 7, nout]
    mlp = Network(nodes, eta, theta, alpha, f)
    mlp.train(x_train, y_train, batch_size, epochs, minJ)

    test_rate = test_network(x_test, y_test, mlp, True)


if __name__ == "__main__":
    main()