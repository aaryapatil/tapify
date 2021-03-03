"""This is the F-test interface.

This computes Thomson F-test using eigencoefficients to
quantify a "pure" periodic signal in coloured noise.
It computes the variance explained by the presence of a
periodic signal and compares it with residual coloured
noise.
"""


__version__ = '0.1'
__author__ = 'Aarya Patil'


# Dependencies
import nfft
import numpy as np

# Local imports
from .transform import ndft
from .tapify import _dpss_interp


def ftest(t, x, NW=4, K=7, tapers=False, center_data=True,
          nyquist_factor=1, method='dft'):
    '''
    Thomson F-test.
    '''
    # Length of time-series
    N = len(t)
    # Total time
    T = t[-1] - t[0]
    # Time sampling
    dt = T/N

    # Center the data - zero mean
    if center_data:
        x -= np.mean(x)

    # Computer interpolated DPSS tapers
    if not tapers:
        tapers, eigvals = _dpss_interp(t, NW, K)

    if method == 'dft':
        fourier = ndft
    elif method == 'fft':
        fourier = nfft.nfft

    # Frequencies for the fourier transform
    freq = (-(N // 2) + np.arange(N))/T
    # Time range -1/2 to 1/2 for fourier transform
    t_ = np.interp(t, (t.min(), t.max()), (-0.5, 0.5))

    taper_data = np.zeros((K, N))
    spec = np.zeros(shape=(K, len(freq)), dtype=np.complex_)

    for ind, window in enumerate(tapers):
        # Normalisation of tapered data
        window = window/np.sum(window**2)*np.sqrt(T/dt)
        # Taper the data
        taper_data[ind] = window*x
        # Eigencoefficients
        spec[ind] = fourier(t_, taper_data[ind])

    # Percival and Walden H0
    tapers = tapers*np.sqrt(dt)
    Uk0 = np.sum(tapers, axis=1)

    # Odd tapers are symmetric and hence their summation goes to 0
    if(K >= 2):
        Uk0[np.arange(1, 7, 2)] = 0

    # H0 sum of squares
    Uk0_sum_sq = np.sum(Uk0**2)

    # Mean estimate of the amplitude of the periodic signal
    num = np.sum(Uk0[:, None]*spec, axis=0)
    mu_f = num/Uk0_sum_sq

    # Variance explained the periodic signal in the data
    sig_var =  np.abs(mu_f)**2*Uk0_sum_sq

    # Variance of residual coloured noise
    noise_var = np.sum(np.abs(spec - Uk0[:, None]*mu_f)**2, axis=0)

    # k-1 degrees of freedom
    f_test = (K-1)*sig_var/noise_var
    return freq, f_test
