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

def f(u):
    return 1 if u > 0 else -1

if __name__ == "__main__":
    dnn = Network(4, np.array([3, 4, 4, 3]), 0.05, 0.1, 10, 20, f)