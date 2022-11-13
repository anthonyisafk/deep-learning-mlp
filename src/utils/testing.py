from classes.network import *


def test_network(x, y, mlp, print_result=True):
    success = 0
    ntests = len(y)
    for tx, ty in zip(x, y):
        res = mlp.predict(tx)
        # print(res)
        pred = np.argmax(res)
        target = np.argmax(ty)
        print(f" -- | {res} | prediction : {pred}, actual y : {target}")
        if pred == target:
            success += 1

    if print_result:
        print(f"\n  >>> Passed {success} / {ntests} tests.")
    return success / ntests
