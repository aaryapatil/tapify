"""
This is the top-level interface that computes the
Multitaper Periodogram.
"""

__version__ = '0.1'
__author__ = 'Aarya Patil'
__all__ = ["MultiTaper"]

import math
import warnings

import numpy as np
import finufft
from scipy import interpolate
from scipy.fft import fft, fftfreq
from scipy.signal.windows import dpss
from astropy import units as u
from astropy.timeseries.periodograms import LombScargle

from .transform import nudft


class MultiTaper():
    """
    Multitaper Periodogram as described in [1].
    Discrete Prolate Spheroidal Sequences (DPSS) tapers are orthogonal
    to each other in the time domain, and are designed to mitigate spectral
    leakage. Using these orthogonal tapers, multiple independent periodograms
    (frequency domain estimates) can be generated. An average of these
    independent periodograms provides the Miltitaper Periodogram which
    minimises both bias and variance.

    Parameters
    ----------
    x : array_like
        Sequence of observations. If ``t`` is provided, these are associated
        with the given times. Else, they are considered to be evenly-sampled.

    t : array_like or `~astropy.units.Quantity`, optional
        Sequence of observation times. Defaults to None for representing
        evenly-sampled data.

    delta_t : float, optional
        Sampling time used in the case of evenly sampled data. This should only
        be provided when ``t`` is None.

    NW : float, optional
        Bandwidth parameter for DPSS

    K : int, optional
        Number of DPSS tapers

    tapers: array_like
        Pre-computed tapers to be used.

    eigvals: array_like
        Eigenvalues of the pre-computed tapers.

    References
    ----------
    .. [1] Thomson, D. J. (1982) "Spectrum estimation and harmonic analysis."
        Proceedings of the IEEE, 70, 1055–1096
    """
    def __init__(self, x, t=None, delta_t=None, NW=4.0, K=7, tapers=None,
                 eigvals=None):

        self.x = x
        self.NW = NW
        self.K = K

        # Length of time-series
        self.N = len(x)

        # Unit associated with time ``t`` will be set if it is a Quantity
        self.unit = None

        if t is not None:
            # The unit conversion here does not affect the user since a
            # Quantity is returned at the end whose units can be changed
            # as desired.
            if isinstance(t, u.Quantity):
                t = t.to_value(u.s)
                self.unit = u.Hz
            self.t = t
            # Total time
            self.T_range = t[-1] - t[0]
            # Time sampling
            self.delta_t = self.T_range/self.N
            # Find out if time samples are equally spaced
            t_diff = np.diff(t)
            if np.all([math.isclose(samp, t_diff[0], rel_tol=1e-10)
                       for samp in t_diff]):
                self.even = True
            else:
                self.even = False
        else:
            # Even sampling case when t is nor provided.
            if delta_t is None:
                raise TypeError('``delta_t`` must be provided when ``t`` \
                                 is unavailable: even-sampling case.')
            self.delta_t = delta_t
            self.T_range = delta_t*self.N
            self.even = True

        if not isinstance(K, int):
            warnings.warn('K should be an integer value. \
                           Float will be rounded to integer.')
            K = int(K)

        # Computer interpolated DPSS tapers
        if not tapers:
            if self.even:
                tapers, eigvals = dpss(self.N, self.NW, self.K,
                                       return_ratios=True)
            else:
                tapers, eigvals = self._dpss_interp()

        self.tapers = tapers
        self.eigvals = eigvals

    def _dpss_interp(self):
        '''
        Computing Discrete Prolate Spheroidal Sequences (DPSS) for
        an irregularly sampled time-series. The DPSS (or Slepian
        Sequences) are sampled regularly using the tridiagonal method
        for accuracy, and then interpolated to the irregular grid
        using spline interpolation.

        DPSS are used as tapers or windows in the multitaper power
        spectral density estimation.

        Returns
        -------
        tapers  : np.ndarray (K, len(t))
            interpolated tapers

        eigvals : np.ndarray (K,)
            eigenvalues for tapers
        '''

        # Evenly sampled DPSS computed using tridiagonal method
        tapers, eigvals = dpss(self.N, self.NW, self.K, return_ratios=True)

        t_even = np.linspace(self.t[0], self.t[-1], self.N)
        tapers_interp = np.asarray(tapers)

        # Interpolate the evenly sampled tapers to given times
        for ind, tp in enumerate(tapers):
            spline = interpolate.splrep(t_even, tp, s=0)
            tapers_interp[ind] = interpolate.splev(self.t, spline, der=0)

        return tapers_interp, eigvals

    def _taper_data(self, center_data=True):
        """
        Taper the sequence of observations.
        """

        # Center the data - zero mean
        if center_data:
            x = self.x - np.mean(self.x)
        else:
            x = self.x

        # Normalisation of tapered data
        tapers = self.tapers.T/np.sum(self.tapers**2, axis=1)
        tapers *= np.sqrt(self.N)
        return tapers.T*x

    def _adaptive_weights(self, powerK, niter=100):
        """
        Adaptive weighting for optimal averaging of tapered periodograms.
        """

        # Initial spectral estimate includes only first two tapers
        spec = np.mean(powerK[:2], axis=0)

        # Could try using sparsity priors to limit iterations - Josh
        # Instaed of running this ``iter`` times, there should be a way to
        # check that the wieghts are optimised - i.e. if the spectrum
        # estimate does not change after a few iterations
        for _ in range(niter):
            # Weighting using local "signal" and broad-band "noise"
            var = np.std(self.x)**2
            weights = spec[:, None]/(spec[:, None]*self.eigvals
                                     + (1-self.eigvals)*var)

            # Spectrum Estimate
            num = np.mean(weights**2*self.eigvals*powerK.T, axis=1)
            den = np.sum(weights**2*self.eigvals, axis=1)
            spec = num/den

        return spec

    def periodogram(self, center_data=True, method='fft', nyquist_factor=1,
                    adaptive_weighting=False, niter=100):
        '''
        Compute the Multitaper Periodogram for even or uneven sampling.

        Parameters
        ----------
        center_data : bool, optional
            If True, center the data by subtracting the mean
            of the observational data. Defaults to True.

        method : string
            Periodogram method. Deafults to Lomb-Scargle

        nyquist_factor : float
            Nyquist factor for setting the frequency limit.
            Deafults to 1, that is, the Nyquist limit.

        adaptive_weighting : bool
            Adaptive weighting to be used or not

        niter : bool
            Number of iterations for adaptive weighting
        '''

        if isinstance(method, str):
            method = method.lower()
        else:
            raise TypeError('``method`` must be a string.')

        if method not in ['dft', 'fft', 'ls']:
            raise ValueError('``method`` must be one of `dft`, `fft`, `ls`')

        x_tapered = self._taper_data(center_data=center_data)

        # Even sampling case - Fourier Transform
        if self.even:
            if method == 'LS':
                warnings.warn('Lomb-Scargle Periodogram is not appropriate \
                               for evenly-sampled data. Classical \
                               Periodogram is used instead.')

            if not nyquist_factor == 1:
                warnings.warn('Nyquist frequency limit is used for \
                               evenly-sampled data: ``nyquist_factor`` = 1')

            # Nyquist frequency limit is (sampling rate)/(2)
            # FUTURE: Use a finer grid for frequency
            freq = fftfreq(self.N, self.delta_t)[:self.N//2]
            powerK = np.zeros((self.K, freq.shape[0]))

            for ind, dat in enumerate(x_tapered):
                # Tapered spectrum estimate using fast LS
                f_t = fft(dat)
                powerK[ind] = np.abs(f_t[0:self.N//2])**2
                # This could be changed to use different normalizations
                # like delta_t * oversample: 0.5*self.delta_t/self.N
                # FUTURE: Add a normalisation argument to ``periodogram``
                powerK[ind] *= 1/self.N

        # Uneven sampling case - Lomb Scargle
        elif method == 'ls':
            # Could replace the astropy LS with scipy LS to reduce dependency,
            # but this will limit the type of algorithm used to compute LS
            freq = LombScargle(self.t, x_tapered[0]).autofrequency(
                nyquist_factor=nyquist_factor)

            powerK = np.zeros((self.K, freq.shape[0]))
            for ind, dat in enumerate(x_tapered):
                # Tapered spectrum estimate using fast LS
                powerK[ind] = LombScargle(self.t, dat).power(freq,
                                                             method='fast')

        # Uneven sampling case - Non-uniform Fourier Transform
        else:
            fourier_type = {'dft': nudft, 'fft': finufft.nufft1d1}.get(method)

            # Covert times to range [-pi to pi] for finufft
            t_scaled = (self.t - self.t[0])*(2*np.pi/self.T_range) - np.pi
            # FUTURE: Tune the frequency grid
            n_freq = nyquist_factor*self.N
            freq = np.arange(n_freq//2)/self.T_range
            powerK = np.zeros((self.K, freq.shape[0]))

            for ind, dat in enumerate(x_tapered):
                # Tapered spectrum estimate using fourier transform
                f_t = fourier_type(t_scaled, dat, n_freq)
                powerK[ind] = np.abs(f_t[n_freq//2:])**2
                powerK[ind] *= 1/self.N

        # Adaptive weighting
        if adaptive_weighting:
            spec = self._adaptive_weights(powerK, niter=niter)
        else:
            # Average to get the multitaper statistic
            spec = np.mean(powerK, axis=0)

        # Convert to quantity is unit is present
        if self.unit:
            freq = u.Quantity(freq, unit=self.unit)
            spec = u.Quantity(spec, unit=1/self.unit)

        return freq, spec

    def ftest(self, center_data=True, nyquist_factor=1, method='fft'):
        '''
        Thomson F-test.
        '''

        # Deal with K = 1 case
        if self.K == 1:
            raise ValueError('K should be greater than 1 for F-test.')

        x_tapered = self._taper_data(center_data=center_data)

        fourier_type = {'dft': nudft, 'fft': finufft.nufft1d1}.get(method)

        # Covert times to range [-pi to pi] for finufft
        t_scaled = (self.t - self.t[0])*(2*np.pi/self.T_range) - np.pi

        # FUTURE: Tune the frequency grid
        n_freq = nyquist_factor*self.N

        # Frequencies for the fourier transform, and the transform
        freq = (-(n_freq//2) + np.arange(n_freq))/self.T_range
        spec = np.zeros(shape=(self.K, len(freq)), dtype=np.complex_)

        for ind, dat in enumerate(x_tapered):
            # Eigencoefficients
            spec[ind] = fourier_type(t_scaled, dat, n_freq)

        # Percival and Walden H0
        tapers = self.tapers*np.sqrt(self.delta_t)
        Uk0 = np.sum(tapers, axis=1)

        # Odd tapers are symmetric and hence their summation goes to 0
        if(self.K >= 2):
            Uk0[np.arange(1, self.K, 2)] = 0

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
        f_test = (self.K-1)*sig_var/noise_var

        if self.unit:
            freq = u.Quantity(freq, unit=self.unit)

        return freq, f_test
