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
from scipy.stats import f
from statsmodels.stats.multitest import multipletests
from astropy import units as u
from astropy.timeseries.periodograms import LombScargle

from .utils import jk_var_helper


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
            raise ValueError(r'N must be greater than 8.')

        if float(NW) < 0.5:
            raise ValueError(r'NW must be greater than or equal to 0.5.')
        if float(NW) > 500:
            warnings.warn(r'NW is greater than 500.', UserWarning)

        if float(NW)/N > 0.5:
            warnings.warn(r'Half-bandwidth parameter W is greater than 0.5.',
                          UserWarning)

        if K < 1:
            raise ValueError(r'K must be greater than or equal to 1.')
        if K > (1.5 + 2*float(NW)):
            warnings.warn(r'K is big compared to 2NW.', UserWarning)
        if not isinstance(K, int):
            warnings.warn(r'K should be an integer value. Float will be '
                          'rounded to integer.', UserWarning)
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
                warnings.warn('``delta_t`` should not be provided when '
                              '``t`` is given. ``delta_t`` will be asigned '
                              'using ``t`` instead.', UserWarning)
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
        tapers *= np.sqrt(self.delta_t)
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
            var = np.sum(spec, axis=-1) / self.N
            bband = (1-eigval_k)*var
            weights = spec[:, None] / (spec[:, None]*eigval_k +
                                       bband)

            # Spectrum Estimate
            num = np.sum(np.abs(weights)**2 * eigval_k * power_k.T, axis=1)
            den = np.sum(np.abs(weights)**2 * eigval_k, axis=1)
            spec_curr = num / den

            # Compute Eq 5.4 in Thomson 1982 to find the recursive solution
            cfn = eigval_k[:, None] * (spec_curr[:, None].T - power_k)
            cfn /= (spec_curr[:, None]*eigval_k + bband).T**2
            cfn = np.sum(cfn, axis=0)
            # Heuristic for comparison
            if np.percentile(cfn**2, 95) < 1e-12:
                break
            else:
                spec = spec_curr

        return spec_curr, weights

    def periodogram(self, center_data=True, method='fft', N_padded='default',
                    nyquist_factor=1, freq=None, adaptive_weighting=False,
                    maxiter=100, ftest=False, jackknife=False, plot=False,
                    alpha=0.05,show_mu_f =False):
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
                warnings.warn('Lomb-Scargle Periodogram is not appropriate '
                              'for evenly-sampled data. Classical Periodogram '
                              'is used instead.', UserWarning)

            if not nyquist_factor == 1:
                warnings.warn('Nyquist frequency limit is used for '
                              'evenly-sampled data: ``nyquist_factor`` '
                              '= 1', UserWarning)

            # Frequencies
            freq = fftfreq(self.N_padded, self.delta_t)
            # Eigenspectra
            spec_k = np.zeros(shape=(self.K, len(freq)), dtype=np.complex128)

            # Positive frequencies
            freq = freq[:self.N_padded//2]
            # Power spectrum estimate
            power_k = np.zeros((self.K, freq.shape[0]))

            for ind, x_k in enumerate(self.x_tapered_padded):
                # Tapered spectrum estimate
                spec_k[ind] = fft(x_k)

            spec_k_pos = spec_k[:, :self.N_padded//2]
            power_k = np.abs(spec_k_pos)**2

        # Uneven sampling case - Lomb Scargle
        elif method == 'ls':
            if freq is None:
                freq = LombScargle(self.t, x_tapered[0]).autofrequency(
                    nyquist_factor=nyquist_factor)
            power_k = np.zeros((self.K, freq.shape[0]))

            for ind, x_k in enumerate(x_tapered):
                # Tapered spectrum estimate using fast LS
                power_k[ind] = LombScargle(self.t, x_k).power(
                    freq, method='fast', normalization='psd')
            power_k = power_k*self.N

            if ftest:
                ftest = False
                warnings.warn('F-test requires complex eigencoefficients '
                              'which Lomb-Scargle Periodogram does not '
                              'provide. Use ``fft`` or ``dft`` instead.',
                              UserWarning)

        # Uneven sampling case - Fourier Transform
        elif method == 'dft' or method == 'fft':
            # Frequencies
            freq = fftfreq(self.N_padded, self.delta_t)
            # Eigenspectra
            spec_k = np.zeros(shape=(self.K, len(freq)), dtype=np.complex128)

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

            spec_k_pos = spec_k[:, self.N_padded//2:]
            power_k = np.abs(spec_k_pos)**2
        else:
            raise ValueError('``method`` must be one of `dft`, `fft`, `ls`')

        # Adaptive weighting
        # ASK: Do we need k eigencoefficients or spectral estimates here
        if adaptive_weighting:
            psd, self.weights = self._adaptive_weights(power_k, self.eigvals,
                                                       maxiter=maxiter)
        else:
            # Average to get the multitaper statistic
            psd = np.mean(power_k, axis=0)

        if jackknife:
            self.jk_var = self._jackknife_variance(
                spec_k_pos, self.eigvals,
                adaptive_weighting=adaptive_weighting,
                ftest_freq=freq, alpha=alpha)

        # Convert to quantity is unit is present
        if self.unit:
            freq = u.Quantity(freq, unit=self.unit)
            psd = u.Quantity(psd, unit=1/self.unit)

        if ftest:
            if show_mu_f:
                fstatistic, mu_f = self._ftest_helper(spec_k_pos, show_mu_f=True)
            else:
                fstatistic = self._ftest_helper(spec_k_pos)
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
            if show_mu_f:
                return freq, psd, fstatistic, mu_f, [spec_k, spec_k_pos, t_scaled, n_freq]
            else:
                return freq, psd, fstatistic
        else:
            return freq, psd

    def _ftest_helper(self, spec_k, indices=None, show_mu_f=False):
        """
        Thomson F-test.
        """
        if indices is not None:
            K = len(indices)
        else:
            K = self.K

        # Deal with K = 1 case
        if K == 1:
            raise ValueError('K should be greater than 1 for F-test.')

        # Percival and Walden H0
        tapers = self.tapers*np.sqrt(self.delta_t)
        Uk0 = np.sum(tapers, axis=1)

        # Odd tapers are symmetric and hence their summation goes to 0
        if(K >= 2):
            Uk0[np.arange(1, K, 2)] = 0

        if indices is not None:
            Uk0 = np.take(Uk0, indices, axis=0)

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
        
        if show_mu_f:
            return f_test, mu_f
        else:
            return f_test

    def _jackknife_variance(self, spec_k, eigval_k, adaptive_weighting=True,
                            ftest_freq=None, alpha=0.05):
        """
        Jackknife variance estimate.

         D.J. Thomson and A.D. Chave, “Jackknifed error estimates for spectra,
         coherences, and transfer functions,” 1991.
        """
        if self.K < 4:
            raise TypeError('K must be a greater than 3 to ensure a stable '
                            'jackknife variance estimate.')

        taper_inds = np.arange(self.K)
        leave1_power = np.empty_like(spec_k, dtype='float64')
        if ftest_freq is not None:
            leave1_ftest = np.empty_like(spec_k, dtype='float64')
            self.leave1_f0 = np.zeros(shape=self.K)

        self.leave1_frej = []
        self.leave1_ftestrej = []

        for ind in range(self.K):
            # Leave-one out eigencoefficients
            leave1_inds = np.delete(taper_inds, ind)
            leave1_spec_k = np.take(spec_k, leave1_inds, axis=0)
            leave1_eigval_k = np.take(eigval_k, leave1_inds)

            # Convert eigencoefficients to power or tapered spectra
            leave1_power_k = np.abs(leave1_spec_k)**2

            # leave-one out multitaper estimates
            if adaptive_weighting:
                leave1_power[ind], _ = self._adaptive_weights(
                    leave1_power_k, leave1_eigval_k)
            else:
                leave1_power[ind] = np.mean(leave1_power_k, axis=0)

            # Leave-one out Ftest estimates
            if ftest_freq is not None:
                leave1_ftest[ind] = self._ftest_helper(leave1_spec_k,
                                                       indices=leave1_inds)
                p_values = 1 - f.cdf(leave1_ftest[ind], 2, 2*self.K-2)
                reject, _ = multipletests(p_values, alpha=alpha,
                                          method='fdr_bh')[:2]
                self.leave1_frej.append(ftest_freq[reject])
                self.leave1_ftestrej.append(leave1_ftest[ind][reject])
                sorted_f = sorted(leave1_ftest[ind])
                self.leave1_f0[ind] = ftest_freq[np.where(
                    leave1_ftest[ind] == sorted_f[-1])]

        # Logarthim of leave-one out multitaper estimates
        leave1_power = np.log(leave1_power)
        jk_var_power = jk_var_helper(self.K, leave1_power)

        # Variance estimate
        # Factor corrected to be not too conservative in
        # D.J. Thomson, “Jackknifing multiple-window spectra,” 1994
        jk_var_power *= (self.K - 1)/(self.K - 1/2)
        return jk_var_power

    def f0_jackknife_variance(self, named_freq, rayleigh=1):
        rayleigh = rayleigh*1/self.t_range
        freq = self.leave1_frej
        ftest = self.leave1_ftestrej

        K = len(freq)
        jk_named_freq = []
        jk_named_ftest = []
        for k in range(K):
            cond = np.where((freq[k] > named_freq - 3*rayleigh)
                            & (freq[k] < named_freq + 3*rayleigh))
            if len(cond[0]) > 0:
                maxf_ind = np.where(ftest[k][cond] == np.max(ftest[k][cond]))
                jk_named_freq.append(freq[k][cond][maxf_ind])
                jk_named_ftest.append(ftest[k][cond][maxf_ind])
        if len(jk_named_freq) > 0:
            return jk_var_helper(len(jk_named_freq), np.array(jk_named_freq))

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
    
    def find_Uk_f(self):
    """
    Ideally this should return the Fourier transform of the tapers, 
    or the Discrete Prolate Spheroidal wavefunctions
    if you have sufficient computational power
    """
    
        Uk_f = [[] for _ in range(3)]
        
        # Scale for nfft (not padded)
        t_scaled_np = (self.t - self.t[0])/self.t_range - 0.5
        
        if self.N % 2:
            n_freq = self.N - 1
        else:
            n_freq = self.N
        
        for ind, x_k in enumerate(self.tapers):
            Uk_f[ind] = nfft.ndft_adjoint(t_scaled_np, x_k, n_freq)
            
        return Uk_f
        
        
