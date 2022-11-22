import numpy as np
import pandas as pd
import time
from sklearn.neighbors import NearestCentroid
from utils.neighbors_centroids import *
from utils.testing import split_trainset_testset

train_fraction = 0.6

# species = {
#     'SEKER': 0.0,
#     'BARBUNYA': 1.0,
#     'BOMBAY': 2.0,
#     'CALI': 3.0,
#     'DERMASON': 4.0,
#     'HOROZ': 5.0,
#     'SIRA': 6.0
# }

species = {
    'Iris-setosa': 0.0,
    'Iris-versicolor': 1.0,
    'Iris-virginica': 2.0,
}
nlabels = len(species)


def main():
    df_train = pd.read_csv("iris/Iris.csv", header=1, delimiter=',')
    # df_train = pd.read_csv("dry-bean/train_dataset.csv", header=1, delimiter=',')
    nrows, ncols = np.shape(df_train)
    x = df_train.values[:, 0:ncols-1]
    y = np.array([species[v] for v in df_train.values[:, ncols-1]])

    samples, targets = split_into_classes(x, y, nlabels, species)
    x_train, x_test, y_train, y_test = split_trainset_testset(samples, targets, train_fraction, species)

    train_start = time.time()
    model = NearestCentroid()
    model.fit(x_train, y_train)
    traintime = time.time() - train_start
    print(f"  >> Fit model for Nearest Centroid Classifier : {traintime:.4f} sec.")

    test_model(model, x_test, y_test, 0)


if __name__ == '__main__':
    main()