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


    def increase_spectral_resolution(self, fs, x_n, t_n):
        pad_factor = 1
        x_n_padded = np.concatenate(x_n,np.zeros(int(len(x_n)*pad_factor)))
        N = len(x_n_padded)

        X_m = np.fft.fft(x_n_padded)

        #freq= np.fft.rfftfreq(N, 1 / fs)

        f_est = np.argmax(X_m)*fs/N

        return f_est, len(t_n)/fs


    def improve_noise_robustness():
        pass

    def estimate_fundamental_frequency_more_reliably():
        pass

    def adapting_to_dynamic_signals():
        pass
