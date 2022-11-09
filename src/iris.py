import csv
import pandas as pd
from network import *


train_fraction = 0.7
minJ = 0.05
alpha = 2.6e-6
theta = 0.0
eta = 0.029
epochs = 100
batch_size = 1
f = activation.tanh

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

    samples = [[] for _ in range(3)]
    targets = [[] for _ in range(3)]
    for v in species.values():
        indices = np.asarray(y[:, v] == 1.0)
        samples[v] = x[indices]
        targets[v] = y[indices]


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

    nin = len(x[0])
    nout = len(y[0])
    nodes = [nin, 7, nout]
    mlp = Network(len(nodes), nodes, eta, theta, alpha, f)

    # for l in mlp.layers:
    #     for i in range(l.n):
    #         print(l.nodes[i])

    mlp.train(x_train, y_train, batch_size, epochs, minJ)

    # for l in mlp.layers:
    #     for i in range(l.n):
    #         print(l.nodes[i])

    success = 0
    ntests = len(y_test)
    for tx, ty in zip(x_test, y_test):
        res = mlp.training_predict(tx)
        # print(res)
        pred = np.argmax(res)
        target = np.argmax(ty)
        print(f" -- | {res} | prediction : {pred}, actual y : {target}")
        if pred == target:
            success += 1
    print(f"\n  >>> Passed {success} / {ntests} tests.")





