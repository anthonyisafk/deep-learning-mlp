"""
Neural Networks & Deep Learning
Aristotle University Thessaloniki - School of Informatics.
******************************
@brief: An *attempt* at making an ANN from scratch. Trained on the MNIST dataset.
@author: Antoniou, Antonios - 9482
@email: aantonii@ece.auth.gr
2022 AUTh Electrical and Computer Engineering.
"""

from network import *
from activation import *
from scipy.io import loadmat
from matplotlib import pyplot

nlayers = 4
nodes_per_layer = np.array([3, 4, 4, 3])
eta = 0.10
theta = 0.0
activf = activation.logistic
batch_size = 1
epochs = 10


if __name__ == "__main__":
    mnist = loadmat("mnist/mnist-original.mat")
    data = mnist['data'].T
    pyplot.plot(data[0])
    pyplot.show()


    # mlp = Network(nlayers, nodes_per_layer, eta, theta, activf)

    # x = [[1, 2, 3], [2, 3, 4], [2, 3, 4.5]]
    # y = [[1, 0, 0], [0, 0, 1], [0, 0, 1]]

    # x = np.concatenate((x, x, x))
    # y = np.concatenate((y, y, y))

    # mlp.train(x, y, batch_size, epochs, 2.1)
