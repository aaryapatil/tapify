# Standard library imports
import random

# Dependencies
import numpy as np
from scipy import interpolate
from scipy.signal.windows import dpss
from astropy import units as u
from astropy.timeseries.periodograms import LombScargle


def dpss_interp(t, NW=4.0, K=7):
    '''
    Computing Discrete Prolate Spheroidal Sequences (DPSS) for
    an irregularly sampled time-series. The DPSS (or Slepian Sequences)
    are sampled regularly using the tridiagonal method for accuracy,
    and then interpolated to the irregular grid using spline
    interpolation.

    DPSS are used as tapers or windows in the multitaper power
    spectral density estimation.

    Parameters
    ----------
    t  : array_like
        times at which model is to be computed

    NW : float, optional
        bandwidth parameter for DPSS

    K  : int, optional
        number of DPSS tapers

    Returns
    -------
    tapers  : np.ndarray (K, len(t))
        interpolated tapers

    eigvals : np.ndarray (K,)
        eigenvalues for tapers
    '''
    # Length of time-series
    N = len(t)

    # Evenly sampled DPSS computed using tridiagonal method
    tapers, eigvals = dpss(N, NW, int(K), return_ratios=True)

    t_even = np.linspace(t[0], t[-1], N)
    tapers_interp = np.asarray(tapers)

    # Interpolate the evenly sampled tapers to given times
    for ind, tp in enumerate(tapers):
        spline = interpolate.splrep(t_even, tp, s=0)
        tapers_interp[ind] = interpolate.splev(t, spline, der=0)

    return tapers_interp, eigvals


def LSspecMT(t, x, NW=4.0, K=7, tapers=None, center_data=False, nyquist_factor=1,
             adaptive_weighting=False, niterations=100):
    '''
    Compute the multitaper Lomb Scargle Periodogram.

    Parameters
    ----------
    t : array_like
        sequence of observation times

    x : array_like
        sequence of observations collected at times t

    NW : float, optional
        bandwidth parameter for DPSS

    K : int, optional
        number of DPSS tapers

    center_data : bool, optional
        if True, pre-center the data by subtracting the mean
        of the observational data
    '''

    if not isinstance(K, int):
        print('K should be an integer value.\
               Float will be rounded to integer.')
        K = int(K)

    if isinstance(t, u.Quantity):
        t = t.to_value(u.s)

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
        tapers, eigvals = dpss_interp(t, NW, K)

    # Taper the data
    taperedData = np.zeros((K, N))

    freq = LombScargle(t, taperedData[0]).autofrequency(nyquist_factor=nyquist_factor)

    powerK = np.zeros((K, freq.shape[0]))

    for ind, window in enumerate(tapers):
        # Normalisation of tapered data
        window = window/np.sum(window**2)*np.sqrt(T/dt)
        taperedData[ind] = window*x
        # Tapered spectrum estimate
        powerK[ind] = LombScargle(t, taperedData[ind]).power(freq, method='fast')

    # Adaptive weighting
    if adaptive_weighting:
        # sparsity priors - idea
        spec = np.mean(powerK[:2], axis=0)
        for i in range(niterations):
            # Weighting using local "signal" and broad-band "noise"
            weights = spec[:, None]/(spec[:, None]*eigvals +
                                     (1-eigvals)*np.std(x)**2)
            # Spectrum Estimate
            num = np.mean(weights**2*eigvals*powerK.T, axis=1)
            den = np.sum(weights**2*eigvals, axis=1)
            spec = num/den
    else:
        # Average to get the multitaper statistic
        spec = np.mean(powerK, axis=0)

    freq = u.Quantity(freq, unit=u.Hz)
    spec = u.Quantity(spec, unit=1/u.Hz)
    return freq, spec