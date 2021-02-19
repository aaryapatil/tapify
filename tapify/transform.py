"""
This interface includes all fourier transforms
that can be utilised for computing periodograms.
"""


import numpy as np


def ndft(x, f):
    """
    Non-equispaced discrete Fourier transform.
    """
    # Length of timeseries
    N = len(x)
    # Evaluate frequencies
    k = -(N // 2) + np.arange(N)
    return np.dot(f, np.exp(2j * np.pi * k * x[:, np.newaxis]))
