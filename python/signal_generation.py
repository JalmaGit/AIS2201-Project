import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from IPython.display import Audio

class signal_generation:

    def __init__(self):
        pass


    def generate_pure_sinewave(fs, N, f, A):
        time = np.arange(N)/fs
        signal = A*np.cos(2*np.pi*f*time) + A/2*np.cos(2*np.pi*f*2*time)

        x_n = signal/max(abs(signal))

        return x_n, time

    def generate_abrubt_sinewave(fs, N, f, A):
        time = np.arange(N)/fs
        signal1 = A*np.sin(2*np.pi*f*time[N//2:])
        signal2 = A*np.cos(2*np.pi*f*1.2*time[:N//2])
        signal = np.concatenate((signal1, signal2))

        x_n = signal/max(abs(signal))
        return  x_n, time

    def get_sound(file_path):
        fs, signal = sp.io.wavfile.read(file_path)

        if (len(signal.shape) == 2):
            signal = signal[:,0]

        x_n = signal/max(abs(signal))               # Scale sample values to the range -1 < x[n] < 1
        N = len(x_n)
        time = np.arange(N)/fs

        return fs, x_n, time, N

    def generate_noisy_signal(signal, SNR_N, Noise_A):
        N = len(signal)

        noisy_signal = np.zeros(N)

        SNR = np.logspace(-2,2, num=SNR_N)
        noise_vars = (((Noise_A)**2)/2)/(1/SNR)

        for index, noise_var in enumerate(noise_vars):
            noise = np.random.normal(scale=np.sqrt(noise_var), size=(N//SNR_N))
            noisy_signal[N//SNR_N*index:N//SNR_N*index+N//SNR_N] = signal[N//SNR_N*index:N//SNR_N*index+N//SNR_N] + noise.copy()

        return noisy_signal