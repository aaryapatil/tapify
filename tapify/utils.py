import numpy as np


def periodogram_slow(data, freq, time=None):
    """
    Slow implementation of the periodogram estimate for evenly sampled
    time series.
    """
    N = data.shape[-1]
    if time is None:
        time = np.arange(0, N)
    dft = np.sum(data * np.exp(-2j * np.pi * freq * time),
                 axis=1)
    return np.abs(dft)**2


def jk_var_helper(K, leave1_est):
    """
    Jackknife variance given leave one estimates
    """
    # Average of the leave one estimates
    leave1_avg_est = leave1_est.mean(axis=0)
    # Average subtracted estimates
    leave1_centered_est = leave1_est - leave1_avg_est
    # Scale factor
    scale_factor = (K - 1)/K
    jk_var = scale_factor * np.sum(leave1_centered_est**2, axis=0)
    return jk_var
