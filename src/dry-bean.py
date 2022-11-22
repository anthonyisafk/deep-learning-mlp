"""
Neural Networks & Deep Learning
Aristotle University Thessaloniki - School of Informatics.
******************************
@brief: An *attempt* at making an ANN from scratch.
        Trained on the `dry bean` dataset.
@author: Antoniou, Antonios - 9482
@email: aantonii@ece.auth.gr
2022 AUTh Electrical and Computer Engineering.
"""

import pandas as pd
from sklearn import preprocessing
from imblearn.over_sampling import SMOTE
from classes.network import *
from utils.testing import *


train_fraction = 0.6
minJ = 0.05
alpha = 3e-5
theta = 0.00
eta = 0.20
epochs = 120
batch_size = 1
f = activation.logistic

species = {
    'SEKER': 0,
    'BARBUNYA': 1,
    'BOMBAY': 2,
    'CALI': 3,
    'DERMASON': 4,
    'HOROZ': 5,
    'SIRA': 6
}
nlabels = len(species)

def main():
    df = pd.read_csv('dry-bean/train_dataset.csv', header=1, delimiter=',')
    nrows = len(df.values[:, 0])
    ncols = len(df.values[0, :])
    x = df.values[:, 1:ncols - 1]
    d = np.zeros(shape=(nrows))
    for i in range(nrows):
        label = species[df.values[i, ncols-1]]
        d[i] = label

    sm = SMOTE()
    x, d = sm.fit_resample(x, d)
    nrows = len(x)
    y = np.zeros(shape=(nrows,nlabels))
    for i in range(nrows):
        y[i, int(d[i])] = 1.0

    samples, targets = split_into_classes(x, y, nlabels, species)
    # frequencies = np.zeros(shape=(nlabels))
    # for i in range(nlabels):
    #     frequencies[i] = len(targets[i])
    # max_freq = np.max(frequencies)
    # scales = [int(np.ceil(max_freq / f)) for f in frequencies]
    # for i in range(nlabels):
    #     targets[i] *= scales[i]
    x_train, x_test, y_train, y_test = split_trainset_testset(samples, targets, train_fraction, species)
    x_train = preprocessing.normalize(x_train)


    # x_test = preprocessing.normalize(x_test)

    nin = len(x[0])
    nout = len(y[0])
    nhidden = int(2 * (nin + nout) / 3)
    nhidden_layer = int(nhidden / 2)
    nodes = [nin, nhidden, nout]
    mlp = Network(nodes, eta, theta, alpha, f)
    mlp.train(x_train, y_train, batch_size, epochs, minJ)

    test_rate = test_network(x_test, y_test, mlp, False)


if __name__ == "__main__":
    main()