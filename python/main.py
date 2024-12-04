from detection import detection_algorithms as da
from signal_generation import signal_generation as sg
import matplotlib.pyplot as plt



fs = 48000
signals = sg()

sampling_freq = 48000
time_in_seconds = 6
samples = sampling_freq*time_in_seconds
frequency = 400
Amplitude = 1

pure_tone, time = sg.generate_pure_sinewave(sampling_freq, samples, frequency, Amplitude)
noisy_pure_tone = sg.generate_noisy_signal(pure_tone,1000, Amplitude*10)
detect = da(fs)

bufferSize = 4096 #Samples in buffer

freq_est = []
time_est = [0]

for index in range(len(noisy_pure_tone)//bufferSize):
    f_est, t = detect.simple_detection(noisy_pure_tone[index*bufferSize:index*bufferSize+bufferSize])

    freq_est.append(f_est)
    time_est.append(t+time_est[-1])

plt.plot(time_est[1:], freq_est)
plt.show()
