import numpy as np

def periodogram_even_slow_impl(data, freq):
    """
    Slow implementation of the periodogram estimate for evenly sampled
    time series.
    """
    N = data.shape[-1]
    time = np.arange(0, N)
    dft = np.sum(data * np.exp(-2j * np.pi * freq * time),
                 axis=1)
    return np.abs(dft)**2