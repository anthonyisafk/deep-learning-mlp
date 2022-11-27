from classes.network import *


def test_network(x, y, mlp, print_result=False):
    success = 0
    ntests = len(y)
    for tx, ty in zip(x, y):
        res = mlp.predict(tx)
        # print(res)
        pred = np.argmax(res)
        target = np.argmax(ty)
        if print_result:
            print(f" -- | {res} | prediction : {pred}, actual y : {target}")
        if pred == target:
            success += 1

    acc = 100 * success / ntests
    print(f"\n  >>> Passed {success} / {ntests} tests. ({acc:2.3f}%)")
    return success / ntests


def split_into_classes(x, y, nlabels, dict=None):
    samples = [[] for _ in range(nlabels)]
    targets = [[] for _ in range(nlabels)]
    if dict is None:
        dict = {i:i for i in range(nlabels)}
    for v in dict.values():
        indices = np.asarray(y[:, v] == 1.0)
        samples[v] = x[indices]
        targets[v] = y[indices]
    return samples, targets


def split_trainset_testset(samples, targets, train_fraction, dict=None):
    x_train = []
    x_test = []
    y_train = []
    y_test = []
    if dict is None:
        dict = {i:i for i in range(len(samples))}
    for v in dict.values():
        iv = int(v)
        nv = len(samples[iv])
        ntrain = int(train_fraction * nv)
        if len(x_train) == 0:
            x_train = samples[iv][0:ntrain]
            x_test = samples[iv][ntrain:]
            y_train = targets[iv][0:ntrain]
            y_test = targets[iv][ntrain:]
        else:
            x_train = np.concatenate((x_train, samples[iv][0:ntrain]))
            x_test = np.concatenate((x_test, samples[iv][ntrain:]))
            y_train = np.concatenate((y_train, targets[iv][0:ntrain]))
            y_test = np.concatenate((y_test, targets[iv][ntrain:]))
    return x_train, x_test, y_train, y_test
