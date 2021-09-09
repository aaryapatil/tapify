import numpy as np


class Spectrum:
    """
    Define the Spectrum of a process and use it for realization of a time-series.
    """
    def __init__(self, N=None, nyquist_freq=None, freq=None, psd=None):
        """
        Initialize frequency, power spectral density, and parameters
        defining corresponding time-series.
        """
        if freq is not None:
            self.freq = freq
            self.M = len(freq)
            # Assume N is even
            self.N = 2*(self.M + 1)
            # Maximum frequency is taken as nyquist
            self.delta_t = 1/(2*freq[-1])
        else:
            # No of fourier modes
            self.N = N
            # Nyquist frequency defines the time sampling
            self.delta_t = 1/(2*nyquist_freq)
            # If N is even, positive fourier modes have indices 1,..., N/2−1
            # Else, mode indices are 1,...,(N-1)/2
            if N % 2 == 0:
                self.M = N/2 - 1
            else:
                self.M = (N - 1)/2

            self.freq = np.arange(1, self.M + 1, dtype='float64')
            self.freq /= self.N*self.delta_t
        
        if psd is not None and len(psd) != len(freq):
            raise ValueError(r'Frequency and power spectral density '
                              'must be of the same length.')
        self.psd = psd
        
        self.flux_lc = None
        self.time_lc = None
        
    def lorentzian_spectrum(self, freq, freq_0=1., hwhm=1., gain=1.):
        """
        Lorentzian spectrum for modeling damped oscillatory
        (quasi-periodic) random processes.
        """
        return gain * hwhm**2 * 1./(hwhm**2 + (freq - freq_0)**2)
    
    def simulate_time_series(self, process='lorentzian', freq=None, psd=None,
                             args=[1., 1., 1.]):
        """
        Simulate time-series given a process or power spectral density.
        """
        if self.psd is None:
            if process == 'lorentzian':
                self.psd = self.lorentzian_spectrum(self.freq, freq_0=args[0],
                                                    hwhm=args[1], gain=args[2])
        
        # Generate two sets of gaussian distributed random numbers
        norm_a = np.random.randn(int(self.M))
        norm_b = np.random.randn(int(self.M))
        
        # Fourier transform of light curve  = SQRT(S/2) * (A + B*i)
        f_w = np.sqrt(0.5*self.psd)*(norm_a + norm_b*1.j)
        
        # For even number of data points, f(w_nyquist) is real -- use only one random number
        if np.mod(self.N, 2) == 0:
            f_w[-1] = np.sqrt(0.5*self.psd[-1])*norm_a[-1]
            
        # Negative frequencies
        f_w_neg = np.conjugate(f_w)
        f_w_neg = f_w_neg[::-1]
        
        # Mean of the timeseries
        mean = 0.0
        
        # Put the mean of the light curve at frequency = 0
        f_w = np.hstack([mean, f_w])
        
        # Add the negative frequencies to the list
        f_w = np.concatenate((f_w, f_w_neg))
        
        freq_w = np.hstack([0, self.freq])
        freq_w = np.concatenate((freq_w, -self.freq[::-1]))
        
        # Take the inverse fourier transform to generate synthetic light curve
        self.flux_lc = np.fft.irfft(f_w, n=self.N)
        self.time_lc = np.arange(1, self.N + 1)*self.delta_t
        
    def plot_time_series(self):
        """
        Plot the generated time-series.
        """
        if self.flux_lc is not None:
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(20, 4))
            plt.plot(self.time_lc, self.flux_lc, lw=0.75, color='black')
            plt.xlabel('Time')
            plt.ylabel('Signal')
    