import os

import pytest
import numpy as np
from numpy.testing import assert_allclose

from tapify import MultiTaper

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


@pytest.fixture
def data(N=500, frequency=10, theta=[8, 2, 5], rseed=0):
    """
    Generate some data for testing
    """
    rng = np.random.RandomState(rseed)
    t = -0.5 + rng.rand(N)
    # Could add sorting to the class to avoid doing it here
    t = np.sort(t)
    tau = 2 * np.pi
    y = theta[0] + theta[1] * np.sin(tau*frequency*t) \
        + theta[2] * np.cos(tau*frequency*t)
    y += 0.5*rng.rand(N)

    return t, y


@pytest.mark.parametrize('method', ['LS', 'dft', 'fft'])
@pytest.mark.parametrize('adaptive_weighting', [True, False])
def test_periodogram(data, method, adaptive_weighting):
    t, y = data

    mt_object = MultiTaper(y, t, NW=1, K=1)

    # Test that multitaper recognises this as an uneven-sampling case
    assert mt_object.even is False

    assert mt_object.N == len(t)
    assert mt_object.T_range == t[-1] - t[0]
    assert mt_object.delta_t == mt_object.T_range/mt_object.N

    freq, power = mt_object.periodogram(method=method,
                                        adaptive_weighting=adaptive_weighting)

    # Test the nyquist limit
    avg_nyquist = 0.5/mt_object.delta_t
    assert_allclose(freq[-1], avg_nyquist, atol=freq[1]-freq[0])

    assert_allclose(freq[np.where(power == np.max(power))], 10,
                    atol=freq[1]-freq[0])
