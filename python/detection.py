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