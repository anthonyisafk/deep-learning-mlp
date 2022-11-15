from classes.network import *
from utils.testing import *
from scipy.io import loadmat


train_fraction = 0.05
minJ = 0.05
alpha = 2e-6
theta = 0.0
eta = 0.03
epochs = 30
batch_size = 1
f = activation.logistic


def main():
    mat = loadmat('usps_all.mat')
    data = mat['data'].T
    nlabels, nsamples, nattr = np.shape(data)
    total_samples = nsamples * nlabels

    x = data[0]
    for i in range(1, nlabels):
        x = np.concatenate((x, data[i]))
    y = np.zeros(shape=(total_samples, nlabels))
    for i in range(nlabels):
        offset = i * nsamples
        for j in range(nsamples):
            y[offset + j, i] = 1.0

    samples, targets = split_into_classes(x, y, nlabels)
    x_train, x_test, y_train, y_test = split_trainset_testset(samples, targets, train_fraction)

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