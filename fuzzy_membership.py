import numpy as np

def s_shaped(x, a, b):
    x = np.asarray(x)
    y = np.zeros_like(x, dtype=float)
    y = np.where(x <= a, 0.0, np.where(x >= b, 1.0, 0.5 * (1 + np.cos(np.pi * (b - x) / (b - a)))))
    return y

def z_shaped(x, a, b):
    return 1.0 - s_shaped(x, a, b)