import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from IPython.display import Audio

class detection_algorithms:

    def __init__(self, fs):
        self.fs = fs

        
    def set_new_fs(self, fs):
        self.fs = fs

    def simple_detection(self, sample):
        N = len(sample)

        Xm = sp.fft.rfft(sample).__abs__()
        #frequencies = sp.fft.rfftfreq(N, 1 / fs)

        f_est = np.argmax(Xm)*self.fs/N

        return f_est, N/self.fs

    def lowpass_filter(self, sample):
        N = len(sample)


    def moving_average(self, sample):
        pass

    def autocorrelation_detection(self, sample):
        N = len(sample)
        autocorr = np.correlate(sample, sample, mode='full')  # Compute full autocorrelation
        autocorr = autocorr[N:]  # Take the second half
        peaks = np.where((autocorr[:-1] < autocorr[1:]) & (autocorr[1:] > autocorr[2:]))[0]

        if len(peaks) > 0:
            f_est = self.fs / peaks[0]
        else:
            f_est = 0  # No clear fundamental frequency

        return f_est, N / self.fs