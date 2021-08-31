import pytest
import numpy as np
from numpy.testing import assert_allclose

from tapify import MultiTaper


@pytest.fixture
def data(N=500, frequency=10, theta=[8, 2, 5], rseed=0):
    """
    Generate some sinusoidal data with white noise for testing.
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
@pytest.mark.parametrize('N_padded', ['default', 500])
def test_periodogram(data, method, adaptive_weighting, N_padded):
    """
    Test multitaper periodogram with different parameters:
    (1) Fourier method for computing periodograms
    (2) Adaptive weighting of eigenspectra
    (3) Zero-padding of time-series for finer frequency resolution.
    """
    t, y = data

    mt_object = MultiTaper(y, t, NW=1, K=1)

    # Test that multitaper recognises this as an uneven-sampling case
    assert mt_object.even is False

    assert mt_object.N == len(t)
    assert mt_object.t_range == t[-1] - t[0]
    assert mt_object.delta_t == mt_object.t_range/mt_object.N

    freq, power = mt_object.periodogram(method=method,
                                        adaptive_weighting=adaptive_weighting,
                                        N_padded=N_padded)

    # Test the nyquist limit
    avg_nyquist = 0.5/mt_object.delta_t
    assert_allclose(freq[-1], avg_nyquist, atol=freq[1]-freq[0])

    assert_allclose(freq[np.where(power == np.max(power))], 10,
                    atol=freq[1]-freq[0])


def test_taper_params():
    """
    Validate multitaper parameters.
    """
    N = 8
    with pytest.raises(ValueError) as err:
        MultiTaper.validate_taper_params(N=N)
    assert err.value.args[0] == ('``N`` must be greater than 8.')

    NW = 0.1
    with pytest.raises(ValueError) as err:
        MultiTaper.validate_taper_params(NW=NW)
    assert err.value.args[0] == ('``NW`` must be greater than or equal '
                                 'to 0.5')
    NW = 600
    with pytest.warns(UserWarning, match='NW is greater than 500.'):
        MultiTaper.validate_taper_params(NW=NW)

    K = 0
    with pytest.raises(ValueError) as err:
        MultiTaper.validate_taper_params(K=K)
    assert err.value.args[0] == ('``K`` must be greater than or equal to 1')

    NW = 4.0
    K = 12
    with pytest.warns(UserWarning, match='``K`` is greater than 1.5 + 2NW'):
        MultiTaper.validate_taper_params(NW=NW, K=K)

    K = 5.5
    with pytest.warns(UserWarning, match='K should be an integer value. '
                                         'Float will be rounded to integer.'):
        MultiTaper.validate_taper_params(K=K)

    N = 10
    NW = 9.5
    with pytest.warns(UserWarning, match='Half-bandwidth parameter (W) is '
                                         'greater than 1/2'):
        MultiTaper.validate_taper_params(N=N, NW=NW)
