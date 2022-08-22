import numpy as np
from astropy.timeseries.periodograms import LombScargle
from scipy import interpolate

from .tapify import MultiTaper
from .utils import periodogram_slow


def optimum_nw(x, t=None, delta_t=None, nw_vals=None,
               method='fft', adaptive_weighting=True):
    """
    Optimal NW (bandwidth) computation using Haley, 2017
    """
    if nw_vals is None:
        nw_vals = np.arange(2, 30, 0.5)

    mse_log_spec = np.zeros(shape=len(nw_vals))

    for nw_ind, NW in enumerate(nw_vals):
        # Set the number of tapers
        K = int(2*NW - 1)

        mt_obj = MultiTaper(x, t=t, delta_t=delta_t, NW=NW, K=K)
        _, _ = mt_obj.periodogram(N_padded=mt_obj.N, method=method,
                                  adaptive_weighting=adaptive_weighting,
                                  jackknife=True)

        W = NW/mt_obj.N

        x_tapered = mt_obj._taper_data()

        if mt_obj.even:
            # Create a grid of frequencies spaced 2W apart
            freq_m = np.arange(0, 1, W/2)
            power_k_m = np.zeros(shape=(K, freq_m.shape[0]))

            for ind_f, freq in enumerate(freq_m):
                power_k_m[:, ind_f] = periodogram_slow(x_tapered,
                                                       freq)
        else:
            # Create a grid of frequencies spaced 2W apart
            # NOTE: Cannot handle freq=0 case with LS
            freq_m = np.arange(W/2, 1, W/2)
            power_k_m = np.zeros(shape=(K, freq_m.shape[0]))

            for ind_k, dat_k in enumerate(x_tapered):
                power_k_m[ind_k] = LombScargle(t, dat_k).power(freq_m)

        spec = np.mean(power_k_m, axis=0)
        spline_fm = interpolate.splrep(freq_m, spec, s=np.sqrt(K))
        spec_dders = interpolate.splev(freq_m, spline_fm, der=2)

        bias = W**4/36*(spec_dders/spec)**2

        var_jk = mt_obj._jackknife_variance(power_k_m, mt_obj.eigvals)

        mse_log_spec[nw_ind] = np.mean(bias + var_jk)

    return mse_log_spec
