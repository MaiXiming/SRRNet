import numpy as np
from scipy import signal

class Filterbank:
    """
    Parameters from Chen et al. 2015: Filter bank canonical correlation analysis for implementing a high-speed SSVEP-based brain–computer interface
    filterbank_num=5 from TDCA / BETA
    """
    def __init__(self, filterbank_num=5, sampling_rate=250):
        self.filterbank_num = filterbank_num
        self.sampling_rate = sampling_rate
        self.fb_weights = np.arange(1,filterbank_num+1)**-1.25 + 0.25
        self.filterbank_setup()


    def filterbank_setup(self):
        passband = np.array([[6, 14, 22, 30, 38, 46, 54, 62, 70, 78], [90] * 10])
        stopband = np.array([[4, 10, 16, 24, 32, 40, 48, 56, 64, 72], [100] * 10])
        rfs = self.sampling_rate / 2

        self.fb_coeffs = []
        for fbii in range(self.filterbank_num):
            wp, ws = np.array([passband[0][fbii], passband[1][fbii]]), np.array([stopband[0][fbii], stopband[1][fbii]])
            Wp, Ws = wp / rfs, ws / rfs
            N, Wn = signal.cheb1ord(Wp, Ws, 3, 40)
            b, a = signal.cheby1(N, 0.5, Wn, 'bandpass')
            self.fb_coeffs.append([b, a])
        
    def fb_filter(self, x):
        ## Filter input signal x into each sub-band and return signals of Nfb bands
        # x (batch, slides, 1, channels, timepoints)
        out = []
        for fbii in range(self.filterbank_num):
            b, a = self.fb_coeffs[fbii][0], self.fb_coeffs[fbii][1]
            # tmp = signal.filtfilt(b, a, x, axis=-1) ## diff betw Matlab & Python
            tmp = signal.filtfilt(b, a, x, axis=-1, padlen=3*(max(len(b),len(a))-1))
            # y[:, :, fbii] = tmp
            out.append(tmp)

        out = np.concatenate(out, axis=-3)
        return out
    
    def filter_subband(self, x, fbii=0):
        ## Filter input signal x into sub-band `fbii`
        b, a = self.fb_coeffs[fbii][0], self.fb_coeffs[fbii][1]
        # tmp = signal.filtfilt(b, a, x, axis=-1)
        out = signal.filtfilt(b, a, x, axis=-1, padlen=3*(max(len(b),len(a))-1))

        return out