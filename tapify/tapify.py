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
import nfft
from scipy import interpolate
from scipy.fft import fft, fftfreq
from scipy.signal.windows import dpss
from astropy import units as u
from astropy.timeseries.periodograms import LombScargle


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

        self.N, self.NW, self.K = MultiTaper.validate_taper_params(N=len(x),
                                                                   NW=NW, K=K)

        self._set_time_params(t, delta_t)

        self.pad = False

        # Compute DPSS tapers
        if not tapers:
            if self.even:
                tapers, eigvals = dpss(self.N, self.NW, self.K,
                                       return_ratios=True)
            else:
                # interpolated tapers for uneven sampling
                tapers, eigvals = self._dpss_interp()

        self.tapers = tapers
        self.eigvals = eigvals
        self.weights = None

    @classmethod
    def validate_taper_params(cls, N=100, NW=4, K=7):
        """
        Validate tapering parameters NW, N and K
        """
        if N < 9:
            raise ValueError('``N`` must be greater than 8.')

        if float(NW) < 0.5:
            raise ValueError('``NW`` must be greater than or equal to 0.5')
        if float(NW) > 500:
            warnings.warn(UserWarning('NW is greater than 500.'))

        if float(NW)/N > 0.5:
            warnings.warn(UserWarning('Half-bandwidth parameter (W) is '
                                      'greater than 1/2'))

        if K < 1:
            raise ValueError('``K`` must be greater than or equal to 1')
        if K > 1.5 + 2*float(NW):
            warnings.warn(UserWarning('``K`` is greater than 1.5 + 2NW'))
        if not isinstance(K, int):
            warnings.warn(UserWarning('K should be an integer value.'
                          'Float will be rounded to integer.'))
            K = int(K)

        return N, NW, K

    def _set_time_params(self, t, delta_t):
        # Unit associated with time ``t`` will be set if it is a Quantity
        self.unit = None

        if t is not None:
            # The unit conversion here does not affect the user since a
            # ``Quantity`` is returned at the end whose units can be changed
            # as desired.
            if isinstance(t, u.Quantity):
                t = t.to_value(u.s)
                self.unit = u.Hz

            # Total time
            self.t_range = t[-1] - t[0]

            # Time sampling
            if delta_t is not None:
                warnings.warn(UserWarning('``delta_t`` should not be provided '
                                          'when ``t`` is given. ``delta_t`` '
                                          'will be asigned using ``t`` '
                                          'instead.'))
            self.delta_t = self.t_range/self.N

            # Find out if time samples are equally spaced
            t_diff = np.diff(t)
            if np.all([math.isclose(samp, t_diff[0], rel_tol=1e-10)
                       for samp in t_diff]):
                self.even = True
            else:
                self.even = False
        else:
            # Even sampling case when t is not provided.
            if delta_t is None:
                raise TypeError('``delta_t`` must be provided when ``t``'
                                'is unavailable: even-sampling case.')

            self.t_range = delta_t*self.N
            self.delta_t = delta_t
            self.even = True

        self.t = t

    def _dpss_interp(self):
        """
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
        """

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

    def _pad_tapered_data(self, x_tapered, N_padded):
        if N_padded == 'default':
            N_padded = int(2 * 2**(np.ceil(np.log2(self.N))))
        elif not isinstance(N_padded, int):
            raise ValueError('``N_padded`` must be "default" or a '
                             'numeric value')
        elif N_padded < self.N:
            raise ValueError('``N_padded`` must be equal to or greater '
                             'than N, the length of the time-series.')

        self.pad = True
        pad_len = N_padded - self.N
        x_tapered_padded = np.pad(x_tapered, ((0, 0), (0, int(pad_len))))

        t_padded = None
        if self.t is not None:
            pad_times = self.t[-1] + np.arange(1, pad_len+1)*self.delta_t
            t_padded = np.concatenate((self.t, pad_times))

        return N_padded, x_tapered_padded, t_padded

    def _adaptive_weights(self, power_k, eigval_k, maxiter=100):
        """
        Adaptive weighting for optimal averaging of tapered periodograms.
        """

        # Initial spectral estimate includes only first two tapers
        spec = np.mean(power_k[:2], axis=0)

        # Run for maximum ``maxiter`` iterations to obtain adaptive weights
        for _ in range(maxiter):
            # Weighting using local "signal" and broad-band "noise"
            var = np.std(self.x)**2
            weights = spec[:, None]/(spec[:, None]*eigval_k +
                                     (1-eigval_k)*var)

            # Spectrum Estimate
            num = np.mean(np.abs(weights)**2*eigval_k*power_k.T, axis=1)
            den = np.sum(np.abs(weights)**2*eigval_k, axis=1)
            spec_curr = num/den

            # Check if successive estimates differ by less than 5%
            error = np.abs((spec_curr-spec)/spec)
            mean_error = np.mean(error)
            if mean_error < 0.05:
                break
            else:
                spec = spec_curr

        return spec_curr, weights

    def periodogram(self, center_data=True, method='fft', N_padded='default',
                    nyquist_factor=1, adaptive_weighting=False, maxiter=100,
                    ftest=False, jackknife=False, plot=False):
        """
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
        """

        if isinstance(method, str):
            method = method.lower()
        else:
            raise TypeError('``method`` must be a string.')

        x_tapered = self._taper_data(center_data=center_data)
        self.N_padded, self.x_tapered_padded, self.t_padded = \
            self._pad_tapered_data(x_tapered, N_padded)

        # Even sampling case - Fourier Transform
        if self.even:
            if method == 'LS':
                warnings.warn(UserWarning('Lomb-Scargle Periodogram is not '
                                          'appropriate for evenly-sampled '
                                          'data. Classical Periodogram is '
                                          'used instead.'))

            if not nyquist_factor == 1:
                warnings.warn(UserWarning('Nyquist frequency limit is used '
                                          'for evenly-sampled data: '
                                          '``nyquist_factor`` = 1'))

            # Frequencies
            freq = fftfreq(self.N_padded, self.delta_t)
            # Eigenspectra
            spec_k = np.zeros(shape=(self.K, len(freq)), dtype=np.complex_)

            # Positive frequencies
            freq = freq[:self.N_padded//2]
            # Power spectrum estimate
            power_k = np.zeros((self.K, freq.shape[0]))

            for ind, x_k in enumerate(self.x_tapered_padded):
                # Tapered spectrum estimate
                spec_k[ind] = fft(x_k)

            spec_k_real = spec_k[:, :self.N_padded//2]
            power_k = np.abs(spec_k_real)**2
            # FUTURE: Add a normalisation argument to ``periodogram``
            power_k *= 1/self.N_padded

        # Uneven sampling case - Lomb Scargle
        elif method == 'ls':
            freq = LombScargle(self.t, x_tapered[0]).autofrequency(
                nyquist_factor=nyquist_factor)
            power_k = np.zeros((self.K, freq.shape[0]))

            for ind, x_k in enumerate(x_tapered):
                # Tapered spectrum estimate using fast LS
                power_k[ind] = LombScargle(self.t, x_k).power(freq,
                                                              method='fast')

            if ftest:
                ftest = False
                warnings.warn(UserWarning('F-test requires complex '
                                          'eigencoefficients which '
                                          'Lomb-Scargle Periodogram does not '
                                          'provide. Use ``fft`` or ``dft`` '
                                          'instead.'))

        # Uneven sampling case - Fourier Transform
        elif method == 'dft' or method == 'fft':
            # Frequencies
            freq = fftfreq(self.N_padded, self.delta_t)
            # Eigenspectra
            spec_k = np.zeros(shape=(self.K, len(freq)), dtype=np.complex_)

            # Positive frequencies
            freq = freq[:self.N_padded//2]
            # Power spectrum estimate
            power_k = np.zeros((self.K, freq.shape[0]))

            if method == 'fft':
                fourier_type = nfft.nfft_adjoint
            elif method == 'dft':
                fourier_type = nfft.ndft_adjoint

            # Time range -1/2 to 1/2 for nfft
            t_range_padded = self.t_padded[-1] - self.t_padded[0]
            t_scaled = (self.t_padded - self.t_padded[0])/t_range_padded - 0.5

            # nfft does not allow N to be odd, so N-1 is used instead of odd N
            if self.N_padded % 2:
                n_freq = self.N_padded - 1
            else:
                n_freq = self.N_padded

            for ind, x_k in enumerate(self.x_tapered_padded):
                # Tapered spectrum estimate using fourier transform
                # Maybe rotate this so next step can be :self.N_padded//2
                spec_k[ind] = fourier_type(t_scaled, x_k, n_freq)

            spec_k_real = spec_k[:, self.N_padded//2:]
            power_k = np.abs(spec_k_real)**2
            power_k *= 1/self.N_padded
        else:
            raise ValueError('``method`` must be one of `dft`, `fft`, `ls`')

        # Adaptive weighting
        if adaptive_weighting:
            psd, self.weights = self._adaptive_weights(power_k, self.eigvals,
                                                       maxiter=maxiter)
        else:
            # Average to get the multitaper statistic
            psd = np.mean(power_k, axis=0)

        if jackknife:
            self.jk_var = self._jackknife_variance(
                power_k, self.eigvals, adaptive_weighting=adaptive_weighting)

        # Convert to quantity is unit is present
        if self.unit:
            freq = u.Quantity(freq, unit=self.unit)
            psd = u.Quantity(psd, unit=1/self.unit)

        if ftest:
            fstatistic = self._ftest_helper(spec_k_real)
        else:
            fstatistic = None

        if plot:
            import matplotlib.pyplot as plt

            if ftest:
                plt, (ax0, ax1) = plt.subplots(2, 1, sharex=True)
                ax0.plot(freq, psd, color='black')
                ax1.plot(freq, fstatistic, color='tab:orange')

                ax0.set_ylabel('Power Spectrum (PSD)')
                ax1.set_ylabel('F-statistic')
            else:
                plt, ax1 = plt.subplots(1, 1)
                ax1.plot(freq, psd, color='black')

                ax1.set_ylabel('Power Spectrum (PSD)')
            if self.unit:
                ax1.set_xlabel(f'Frequency {self.unit}')
            else:
                ax1.set_xlabel('Frequency')
            plt.show()

        self.freq = freq
        self.psd = psd
        self.fstatistic = fstatistic

        if ftest:
            return freq, psd, fstatistic
        else:
            return freq, psd

    def _ftest_helper(self, spec_k):
        """
        Thomson F-test.
        """
        # Deal with K = 1 case
        if self.K == 1:
            raise ValueError('K should be greater than 1 for F-test.')

        # Percival and Walden H0
        tapers = self.tapers*np.sqrt(self.delta_t)
        Uk0 = np.sum(tapers, axis=1)

        # Odd tapers are symmetric and hence their summation goes to 0
        if(self.K >= 2):
            Uk0[np.arange(1, self.K, 2)] = 0

        # H0 sum of squares
        Uk0_sum_sq = np.sum(Uk0**2)

        # Mean estimate of the amplitude of the periodic signal
        num = np.sum(Uk0[:, None]*spec_k, axis=0)
        mu_f = num/Uk0_sum_sq

        # Variance explained the periodic signal in the data
        sig_var = np.abs(mu_f)**2*Uk0_sum_sq
        # Variance of residual coloured noise
        noise_var = np.sum(np.abs(spec_k - Uk0[:, None]*mu_f)**2, axis=0)

        # k-1 degrees of freedom
        f_test = (self.K-1)*sig_var/noise_var

        return f_test

    def _jackknife_variance(self, power_k, eigval_k, adaptive_weighting=True):
        """
        Jackknife variance estimate.

         D.J. Thomson and A.D. Chave, “Jackknifed error estimates for spectra,
         coherences, and transfer functions,” 1991.
        """

        taper_inds = np.arange(self.K)
        leave1_est = np.empty_like(power_k)

        for ind in range(self.K):
            # Leave-one out multitaper estimate
            leave1_power_k = np.take(power_k, np.delete(taper_inds, ind),
                                     axis=0)
            leave1_eigval_k = np.take(eigval_k, np.delete(taper_inds, ind))

            if adaptive_weighting:
                leave1_est[ind], _ = self._adaptive_weights(
                    leave1_power_k, leave1_eigval_k)
            else:
                leave1_est[ind] = np.mean(leave1_power_k, axis=0)

        # Logarthim of leave-one out estimates
        leave1_est = np.log(leave1_est)
        # Average of the log estimates
        leave1_avg_est = leave1_est.mean(axis=0)
        # Average subtracted estimates
        leave1_centered_est = leave1_est - leave1_avg_est

        # Variance estimate
        # Factor corrected to be not too conservative in
        # D.J. Thomson, “Jackknifing multiple-window spectra,” 1994
        factor = (self.K - 1)**2/(self.K*(self.K - 1/2))
        jk_var = factor * np.sum(leave1_centered_est**2, axis=0)

        return jk_var

    def efficiency(self, k=None):
        """
        Stability of a multitaper estimate v(f) and the average
        over K multitaper estimates v(f)/2K.

        This is intimately related to bias. If v(f)/2K << 1,
        then either W is too small, or prewhitening is needed.

        Parameters
        ----------
        k : int, optional
            Defaults to K.
        """
        if k is None:
            k = self.K
        elif k > self.K:
            raise ValueError(f'``k`` must be smaller than the total number \
                               of tapers, {self.K}')

        sum_taperssq = 1/k*np.sum(self.tapers[:k, :]**2, axis=0)
        self.var_eff = 1/(self.N*np.sum(sum_taperssq**2))

        if self.weights is None:
            raise ValueError('Adaptive weights required for efficiency.')
        else:
            self.v = 2*np.sum(np.abs(self.weights[:, :k])**2, axis=1)
            self.v_avg = np.mean(self.v)/(2*k)
            self.overall_eff = self.v_avg*self.var_eff*self.N
