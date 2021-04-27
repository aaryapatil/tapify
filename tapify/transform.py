"""
This interface includes all fourier transforms
that can be utilised for computing periodograms.
"""


import numpy as np


def nudft(x, f, n_modes):
    """
    Non-equispaced discrete Fourier transform.
    """
    k = -(n_modes//2) + np.arange(n_modes)
    return np.dot(f, np.exp(1j*k*x[:, np.newaxis]))
