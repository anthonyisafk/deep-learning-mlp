"""
Neural Networks & Deep Learning
Aristotle University Thessaloniki - School of Informatics.
******************************
@brief: An *attempt* at making an ANN from scratch.
        Trained on the `white wine quality` dataset.
@author: Antoniou, Antonios - 9482
@email: aantonii@ece.auth.gr
2022 AUTh Electrical and Computer Engineering.
"""


import pandas as pd
from classes.network import *
from utils.testing import *


train_fraction = 0.6
minJ = 0.05
alpha = 1e-4
theta = 0.0
eta = 0.05
epochs = 20
batch_size = 1
f = activation.logistic

labels = {
    3: 0,
    4: 1,
    5: 2,
    6: 3,
    7: 4,
    8: 5,
    9: 6
}


def main():
    df = pd.read_csv('wine-quality/winequality-white.csv', header=0, delimiter=';')
    nrows = len(df.values[:, 0])
    ncols = len(df.values[0, :])
    nlabels = len(labels)

    x = df.values[:, 0:ncols - 1]
    y = np.zeros(shape=(nrows,nlabels))
    for i in range(nrows):
        y[i, labels[df.values[i, ncols-1]]] = 1.0

    samples, targets = split_into_classes(x, y, nlabels, labels)
    frequencies = np.zeros(shape=(nlabels))
    for i in range(nlabels):
        frequencies[i] = len(targets[i])
    max_freq = np.max(frequencies)
    ncopies = [int(np.ceil(max_freq / f)) - 1 for f in frequencies]
    for i in range(nlabels):
        scopy = samples[i]
        tcopy = targets[i]
        for _ in range(ncopies[i]):
            samples[i] = np.concatenate((samples[i], scopy))
            targets[i] = np.concatenate((targets[i], tcopy))
    x_train, x_test, y_train, y_test = split_trainset_testset(samples, targets, train_fraction, labels)



    nin = len(x[0])
    nout = len(y[0])
    total_hidden = 2 * nin / 3 + nout
    per_hidden_layer = int(total_hidden / 2)
    nodes = [nin, per_hidden_layer, per_hidden_layer, nout]
    mlp = Network(nodes, eta, theta, alpha, f)
    mlp.train(x_train, y_train, batch_size, epochs, minJ)

    test_rate = test_network(x_test, y_test, mlp, True)


if __name__ == '__main__':
    main()
