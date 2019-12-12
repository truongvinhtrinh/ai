import numpy as np


def norm_1(x):
    try:
        return np.sum(np.abs(x))
    except Exception as e:
        return print(e)


y = np.array([4, 1, 3])
print(y.argmax())
print(y.argmin())

