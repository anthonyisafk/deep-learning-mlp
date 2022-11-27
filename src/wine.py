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
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import normalize


train_fraction = 0.6
# minJ = 0.05
minJ = 0.10 
alpha = 0.0001
theta = 0.0
eta = 0.001
epochs = 100
batch_size = 1
f = [activation.logistic, activation.logistic, activation.logistic]

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

    # x = df.values[:, 0:ncols - 1]
    # d = np.zeros(shape=(nrows))
    # for i in range(nrows):
    #     d[i] = int(labels[df.values[i, ncols-1]])

    # sm = SMOTE(k_neighbors=2)
    # x, d = sm.fit_resample(x, d)
    # nrows = len(x)
    # y = np.zeros(shape=(nrows, nlabels))
    # for i in range(nrows):
    #     y[i, int(d[i])] = 1.0
    # x = normalize(x, norm="l2")

    n = 100
    x = np.zeros(shape=(2*n,10))
    y = np.zeros(shape=(2*n,2))
    nlabels = 2
    for i in range(n):
        y[i, 0] = 1.0
    for i in range(n,2*n):
        y[i, 1] = 1.0
    for i in range(n):
        x[i, :] = np.random.normal(loc=-2.0, scale=0.5)
    for i in range(n,2*n):
        x[i, :] = np.random.normal(loc=-0.5, scale=0.5)

    perm = np.random.permutation(2*n)
    x, y = x[perm], y[perm]

    # samples, targets = split_into_classes(x, y, nlabels, labels)
    # x_train, x_test, y_train, y_test = split_trainset_testset(samples, targets, train_fraction, labels)
    samples, targets = split_into_classes(x, y, nlabels, None)
    x_train, x_test, y_train, y_test = split_trainset_testset(samples, targets, train_fraction, None)

    nin = len(x[0])
    nout = len(y[0])
    total_hidden = 2 * nin / 3 + nout
    # per_hidden_layer = int(total_hidden / 2)
    nodes = [nin, nin + nout, nin + nout, nout]
    mlp = Network(nodes, eta, theta, alpha, f)
    mlp.train(x_train, y_train, batch_size, epochs, minJ)

    test_rate = test_network(x_test, y_test, mlp)


if __name__ == '__main__':
    main()
