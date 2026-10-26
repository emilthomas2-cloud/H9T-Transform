"""Explicit normalized nine-point Helmert transform, and tensor powers."""
import numpy as np


def helmert9():
    """Rows: normalized global mean, then eight Helmert contrasts."""
    a = np.zeros((9, 9), dtype=np.float64)
    a[0] = 1 / 3
    for k in range(1, 9):
        scale = np.sqrt(k * (k + 1))
        a[k, :k] = 1 / scale
        a[k, k] = -k / scale
    return a


def transform(x, inverse=False):
    """Apply A9 tensor-power to flattened length 9**L, no dense matrix.

    The convention is a 1D flattened tensor and applying A9 along every axis.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1 or x.size < 9:
        raise ValueError('Expected one-dimensional vector with length 9**L, L>=1')
    n, level = x.size, 0
    while n > 1 and n % 9 == 0:
        n //= 9
        level += 1
    if n != 1:
        raise ValueError('Length must equal 9**L')
    a = helmert9().T if inverse else helmert9()
    y = x.reshape((9,) * level).copy()
    for axis in range(level):
        y = np.moveaxis(np.tensordot(a, y, axes=(1, axis)), 0, axis)
    return y.ravel()


def reconstruction_error(x):
    x = np.asarray(x, dtype=np.float64)
    return float(np.linalg.norm(transform(transform(x), inverse=True)-x) / np.linalg.norm(x))
