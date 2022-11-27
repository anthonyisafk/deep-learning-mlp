import numpy as np
import pandas as pd
import time
from sklearn.neighbors import KNeighborsClassifier
from utils.neighbors_centroids import *
from utils.testing import split_trainset_testset

train_fraction = 0.8

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

    knn1_start = time.time()
    knn1 = KNeighborsClassifier(n_neighbors=1, algorithm='auto')
    knn1.fit(x_train, y_train)
    knn1_traintime = time.time() - knn1_start
    print(f"  >> Fit model for number of neighbors : 1 [{knn1_traintime:.4f} sec.]")

    knn3_start = time.time()
    knn3 = KNeighborsClassifier(n_neighbors=3, algorithm='auto')
    knn3.fit(x_train, y_train)
    knn3_traintime = time.time() - knn3_start
    print(f"  >> Fit model for number of neighbors : 3 [{knn3_traintime:.4f} sec.]")
    print()

    test_model(knn1, x_test, y_test, 1)
    test_model(knn3, x_test, y_test, 3)


if __name__ == '__main__':
    main()