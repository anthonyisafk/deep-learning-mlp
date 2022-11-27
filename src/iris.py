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
import logging
from sklearn.preprocessing import normalize
from classes.network import *
from utils.testing import *

train_fraction = 0.6
minJ = 0.05
min_acc = 0.95
alpha = 0.0005
theta = 0.0
eta = 0.05
epochs = 100
batch_size = 1
f = [activation.logistic, activation.logistic]
f1 = 'l'
f2 = 'l'

logname = "acc95.log"
logging.basicConfig(
    filename=logname,
    filemode='a',
    level=logging.INFO,
    format='%(message)s'
)

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
    x = normalize(x)

    samples, targets = split_into_classes(x, y, 3, species)
    x_train, x_test, y_train, y_test = split_trainset_testset(samples, targets, train_fraction, species)

    nin = len(x[0])
    nout = len(y[0])
    nodes = [nin, nin + nout, nout]
    mlp = Network(nodes, eta, theta, alpha, f)
    training_acc = mlp.train(x_train, y_train, batch_size, epochs, minJ, min_acc)

    testing_acc = test_network(x_test, y_test, mlp, False)

    logging.info(f"{eta},{alpha},{f1},{f2},{training_acc:.3},{testing_acc:.3}")


if __name__ == "__main__":
    main()