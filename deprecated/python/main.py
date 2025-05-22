from detection import detection_algorithms as da
from signal_generation import signal_generation as sg
import matplotlib.pyplot as plt
from IPython.display import Audio

def test(signal, fs, title, bufferSize = 1024):
    detect = da()

    time_est = [0]
    freq_est = []   
    freq_est1 = []
    freq_est2 = []
    freq_est3 = []

    # Tuning Params: 
    ## Padding Factor
    pad_fact = bufferSize
    
    ## Sliding Window:
    windows_size = 0.1
    step_size = 0.05

    ## HPS 
    harmonics = 2

    for index in range(len(signal)//bufferSize):
        x_n = signal[index*bufferSize:index*bufferSize+bufferSize]
        filtered_signal = detect.low_pass(fs, x_n)
        spectraled_signal = detect.increase_spectral_resolution(filtered_signal, pad_fact)
        f_est = detect.simple_detection(fs, spectraled_signal)
        f_est1 = detect.autocorrelation(fs,spectraled_signal)
        f_est2 = detect.harmonic_product_spectrum(fs, spectraled_signal, harmonics)

        t = len(x_n)/fs

        freq_est.append(f_est)
        freq_est1.append(f_est1)
        freq_est2.append(f_est2)
        time_est.append(t+time_est[-1])

    #print(f"{freq_est[0] = }, {freq_est1[0] = },{freq_est2[0] = }")

    plt.plot(time_est[1:], freq_est, label="Simple", linewidth=2)
    plt.plot(time_est[1:], freq_est1, label="Auto Correlation", linewidth=2)
    plt.plot(time_est[1:], freq_est2, label="HPS", linewidth=2)
    plt.title(title)
    plt.legend()
    plt.show()


fs = 48000
signals = sg()

sampling_freq = 48000
time_in_seconds = 6
samples = sampling_freq*time_in_seconds
frequency = 200
Amplitude = 1

pure_tone, time = sg.generate_pure_sinewave(sampling_freq, samples, frequency, Amplitude)
noisy_pure_tone = sg.generate_noisy_signal(pure_tone,1000, Amplitude*2)

test(noisy_pure_tone, sampling_freq, "Pure Sinus")

harmonic_tone, time = sg.generate_harmonic_sinewave(sampling_freq, samples, frequency, Amplitude)
noisy_harmonic_tone = sg.generate_noisy_signal(harmonic_tone,1000, Amplitude*2)

test(noisy_harmonic_tone, sampling_freq, "Harmonic Tone")

abrupt_change, time = sg.generate_abrubt_sinewave(sampling_freq, samples, frequency, Amplitude)

test(abrupt_change, sampling_freq, "Abrupt Tone")

sampling_freq, farm_noise, time, N = sg.get_sound("python/data/Zauberflöte_vocal.wav")

test(farm_noise, sampling_freq, "Zauberflöte_vocal")

sampling_freq, farm_noise, time, N = sg.get_sound("python/data/B_oboe.wav")

test(farm_noise, sampling_freq, "B_oboe")

sampling_freq, farm_noise, time, N = sg.get_sound("python/data/BackNoiseFarm.wav")

test(farm_noise, sampling_freq, "BackNoiseFarm")