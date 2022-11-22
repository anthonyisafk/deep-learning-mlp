import numpy as np


def split_into_classes(x, y, nlabels, dict=None):
    samples = [[] for _ in range(nlabels)]
    targets = [[] for _ in range(nlabels)]
    if dict is None:
        dict = {i:i for i in range(nlabels)}
    for v in dict.values():
        indices = np.where(y == v)[0]
        samples[int(v)] = x[indices]
        targets[int(v)] = y[indices]
    return samples, targets


def test_model(model, x_test, y_test, n_neighbors):
    success = 0
    ntests = len(x_test)
    for i in range(ntests):
        pred = model.predict([x_test[i]])
        if pred == y_test[i]:
            success += 1
    acc = 100 * success / ntests
    if n_neighbors > 0:
        print(f"   >>> KNeighborsClassifier with n_neighbors={n_neighbors} : {acc:2.3f} accuracy.")
    else:
        print(f"   >>> NearestCentroidClassifier : {acc:2.3f} accuracy.")
