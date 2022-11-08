import csv
import pandas as pd
from network import *


train_fraction = 0.9
minJ = 0.05
theta = 0.0
eta = 0.10
epochs = 20
batch_size = 1
f = activation.logistic

species = {
    'Iris-setosa': 0,
    'Iris-versicolor': 1,
    'Iris-virginica': 2,
}

if __name__ == "__main__":
    df = pd.read_csv('iris/iris.csv', header=0)
    nrows = len(df.values[:, 0])
    ncols = len(df.values[0, :])
    x = df.values[:, 1:ncols - 1]
    y = np.zeros(shape=(nrows,3))
    for i in range(nrows):
        y[i, species[df.values[i, ncols-1]]] = 1.0

    perm = np.random.permutation(nrows)
    x, y = x[perm], y[perm]
    ntrain = int(train_fraction * nrows)
    ntest = nrows - ntrain
    x_train, x_test = x[0:ntrain], x[ntrain:nrows]
    y_train, y_test = y[0:ntrain], y[ntrain:nrows]

    nin = len(x[0])
    nout = len(y[0])
    nodes = [nin, 4, 3, nout]
    mlp = Network(len(nodes), nodes, eta, theta, f)

    # for l in mlp.layers:
    #     for i in range(l.n):
    #         print(l.nodes[i])

    mlp.train(x_train, y_train, batch_size, epochs, minJ)

    # for l in mlp.layers:
    #     for i in range(l.n):
    #         print(l.nodes[i])

    success = 0
    for tx, ty in zip(x_test, y_test):
        res = mlp.training_predict(tx)
        # print(res)
        pred = np.argmax(res)
        target = np.argmax(ty)
        print(f" -- prediction : {pred}, actual y : {target}")
        if pred == target:
            success += 1
    print(f"\n  >>> Passed {success} / {ntest} tests.")






