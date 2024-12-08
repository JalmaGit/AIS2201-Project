from detection import detection_algorithms as da
from signal_generation import signal_generation as sg
import matplotlib.pyplot as plt
import numpy as np


fs = 48000
signals = sg()

sampling_freq = 48000
time_in_seconds = 6
samples = sampling_freq*time_in_seconds
frequency = 400
Amplitude = 1

pure_tone, time = sg.generate_pure_sinewave(sampling_freq, samples, frequency, Amplitude)
noisy_pure_tone = sg.generate_noisy_signal(pure_tone,1000, Amplitude)
detect = da()

time_est = [0]
freq_est = []   
freq_est1 = []
freq_est2 = []

bufferSize = 4096 #Samples in buffer
for index in range(len(noisy_pure_tone)//bufferSize):
    x_n = noisy_pure_tone[index*bufferSize:index*bufferSize+bufferSize]
    filtered_signal = detect.low_pass(fs, x_n)
    spectraled_signal = detect.increase_spectral_resolution(x_n, 2)
    f_est = detect.simple_detection(fs, spectraled_signal)
    f_est1 = detect.frequency_domain_autocorrelation(fs, spectraled_signal)
    f_est2 = detect.harmonic_product_spectrum(fs,spectraled_signal, 3)

    t = len(x_n)/fs

    freq_est.append(f_est)
    freq_est1.append(f_est1)
    freq_est2.append(f_est2)
    time_est.append(t+time_est[-1])

print(f"{freq_est[10] = }, {freq_est1[10] = },{freq_est2[10] = }")

plt.plot(time_est[1:], freq_est, label="simple")
plt.plot(time_est[1:], freq_est1, label="auto")
plt.plot(time_est[1:], freq_est2, label="HPS")
plt.legend()
plt.show()

freq_est = []
time_est = [0]
"""
bufferSize = 4096 #Samples in buffer
for index in range(len(noisy_pure_tone)//bufferSize):
    filtered_signal = detect.low_pass(fs, noisy_pure_tone[index*bufferSize:index*bufferSize+bufferSize])
    spectraled_signal = detect.increase_spectral_resolution(filtered_signal)
    f_est, t = detect.simple_detection(fs, spectraled_signal)
        
    freq_est.append(f_est)
    time_est.append(t+time_est[-1])


plt.plot(time_est[1:], freq_est)
plt.show()
"""