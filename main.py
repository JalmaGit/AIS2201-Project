import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from IPython.display import Audio
from tasks import frameworks, modifications


def test(signal, fs, title, one_singal = 48000*4, loops = 12, bufferSize = 2048):
    detect = modifications.detection()
    mods = modifications.preprocessing(bufferSize)

    time_est = [0]
    freq_est = []   
    freq_est1 = []
    freq_est2 = []
    freq_est3 = []
    freq_est4 = []
    freq_est5 = []
    freq_est6 = []
    freq_est7 = []
    freq_est8 = []

    # Tuning Params: 
    ## Padding
    delta_f = 1 #Hz

    ## BandPass Filter:
    high_cut = 20 #Hz
    low_cut = 4200 #Hz
    taps = 129
    window = "chebychev"

    ## Sliding Buffer
    size = 1024

    ## Moving Average
    size_average = 6

    ## HPS 
    harmonics = 3
    mode = "product"

    run_once = 0

    spec = None
    non = None

    for index in range(len(signal)//bufferSize):
        x_n = signal[index*bufferSize:index*bufferSize+bufferSize]

        if run_once == len(signal)//bufferSize:
            non = x_n.copy()
            x_n_p = mods.zero_padding(x_n,delta_f,fs)
            spec = mods.band_pass_filter(x_n_p, low_cut, high_cut, taps, window, fs)
            non = mods.moving_average_window(x_n_p,size_average,window,fs)
            run_once = False

        f_est = detect.simple_detection(x_n, fs)
        x_n_p = mods.zero_padding(x_n,delta_f,fs)
        f_est1 = detect.simple_detection(x_n_p, fs)
        #f_est1 = detect.auto_correlation_pitch(x_n, fs)
        #f_est2 = detect.harmonic_spectrum(x_n,  harmonics , mode, fs)

        x_n_buf = mods.sliding_buffer(x_n,size, fs)
        #f_est3 = detect.simple_detection(x_n_buf, fs)
        #f_est4 = detect.auto_correlation_pitch(x_n_buf, fs)
        #f_est5 = detect.harmonic_spectrum(x_n_buf,  harmonics , mode, fs)

        x_n_p = mods.zero_padding(x_n_buf,delta_f,fs)
        x_n_f = mods.band_pass_filter(x_n_p, low_cut, high_cut, taps, window, fs)

        f_est6 = detect.simple_detection(x_n_f, fs)
        f_est7 = detect.auto_correlation_pitch(x_n_f, fs)
        f_est8 = detect.harmonic_spectrum(x_n_f,  harmonics , mode, fs)



        t = len(x_n)/fs

        freq_est.append(f_est)
        freq_est1.append(f_est1)
        #freq_est2.append(f_est2)
        #freq_est3.append(f_est3)
        #freq_est4.append(f_est4)
        #freq_est5.append(f_est5)
        #freq_est6.append(f_est6)
        #freq_est7.append(f_est7)
        #freq_est8.append(f_est8)
        time_est.append(t+time_est[-1])

    side = False

    if side == True:
        SNR = np.logspace(-2, 4, len(freq_est8))
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharex=True, sharey=True)
        fig.suptitle(title)

        # Simple Estimation
        axes[0].plot(SNR, freq_est6)
        axes[0].set_xscale('log')
        axes[0].set_yscale('log')
        axes[0].set_title('Simple Estimation')
        axes[0].set_xlabel('1/SNR')
        axes[0].set_ylabel('Frequency Estimate')
        axes[0].grid()

        # Auto Correlation
        axes[1].plot(SNR, freq_est7)
        axes[1].set_xscale('log')
        axes[1].set_yscale('log')
        axes[1].set_title('Auto Correlation')
        axes[1].set_xlabel('1/SNR')
        axes[1].grid()

        # Harmonic Product Spectrum
        axes[2].plot(SNR, freq_est8)
        axes[2].set_xscale('log')
        axes[2].set_yscale('log')
        axes[2].set_title('Harmonic Product Spectrum')
        axes[2].set_xlabel('1/SNR')
        axes[2].grid()

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()


    binding = False
    
    print(len(freq_est))

    if binding == True:
        SNR = np.logspace(-2, 4, len(freq_est))

        plt.figure()
        plt.title(title)
        #plt.plot(SNR, freq_est, label='No Padding')
        #plt.plot(SNR, freq_est1, label='Padding')
        #lt.magnitude_spectrum(spec, fs)
        plt.magnitude_spectrum(non, fs)
        #plt.xscale('log')
        plt.xlabel('1/SNR')
        plt.ylabel('Frequency Estimate')
        plt.legend()
        plt.grid()
        plt.show()

    normal = True

    if normal == True:
        SNR = np.logspace(-2, 4, len(freq_est8))

        plt.figure()
        plt.title(title)
        plt.plot(SNR, freq_est7, label='Auto Correlation')
        plt.plot(SNR, freq_est6, label='Simple Estimation')
        plt.plot(SNR, freq_est8, label='Harmonic Product Spectrum')
        plt.yscale("log")
        plt.xscale('log')

        plt.xlabel('1/SNR')
        plt.ylabel('Frequency Estimate')
        plt.legend()
        plt.grid()
        plt.show()

    enable = False

    if enable == True:

        SNR = np.logspace(-2, 4, num=loops)
        print(len(freq_est))
        print(len(signal)//bufferSize)
        print(one_singal//bufferSize)
        print((len(signal)//bufferSize)//(one_singal//bufferSize))

        window_length = one_singal // bufferSize
        iterations    = (len(signal)//bufferSize) // (one_singal//bufferSize)

        # split each block into N sub‐windows:
        N = 6
        sub_length = window_length // N

        comparisons = {
            "Simple(baseline) vs Simple(sliding)":   (np.array(freq_est),  np.array(freq_est3),np.array(freq_est6)),
            "AutoCorr(baseline) vs AutoCorr(sliding)": (np.array(freq_est1), np.array(freq_est4),np.array(freq_est7)),
            "HPS (baseline) vs HPS (sliding)":       (np.array(freq_est2), np.array(freq_est5),np.array(freq_est8)),
        }

        # 2) compute mean & std for each leg of each comparison, on sub‐windows
        stats = {}
        for name, (A, B, C) in comparisons.items():
            stats[name] = {}
            for tag, arr in zip(("A", "B", "C"), (A, B, C)):
                means, stds = [], []
                for i in range(iterations):
                    block = arr[i*window_length:(i+1)*window_length]
                    # subdivide into N sub‐windows:
                    for j in range(0, window_length, sub_length):
                        sub = block[j : j + sub_length]
                        means.append(np.mean(sub))
                        stds .append(np.std(sub))
                stats[name][tag] = {
                    "mean": np.array(means),
                    "std":  np.array(stds)
                }

            # redefine your labels properly:
        tag_labels = {
            "A": "Baseline",
            "B": "Sliding",
            "C": "Sliding with Padding"
        }

        # choose one style per series:
        styles = {
            "A": "-",
            "B": "--",
            "C": ":"
        }

        fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharex='col')
        fig.suptitle(title, fontsize=18)

        # row labels
        axes[0,0].set_ylabel('Mean Frequency (Hz)')
        axes[1,0].set_ylabel('Absolute Error (Hz)')
        for ax in axes[1]:
            ax.set_xlabel('1/SNR')

        for col, (name, comp) in enumerate(stats.items()):
            ax_mean  = axes[0, col]
            ax_error = axes[1, col]
            ax_mean.set_title(name)

            # recompute SNR_sub for this column
            M       = comp["A"]["mean"].shape[0]
            SNR_sub = np.logspace(-2, 4, num=M)

            # now loop over all three tags A, B, C
            for tag in ("A", "B", "C"):
                m = comp[tag]["mean"]
                s = comp[tag]["std"]
                style = styles[tag]
                label = tag_labels[tag]

                # mean ±σ
                ax_mean.set_xscale('log')
                ax_mean.set_yscale('log')
                ax_mean.plot(SNR_sub, m, style, lw=2, label=label)
                ax_mean.fill_between(SNR_sub, m - s, m + s, alpha=0.3)

                # error
                err = np.abs(f - m)
                ax_error.set_xscale('log')
                ax_error.set_yscale('log')
                ax_error.plot(SNR_sub, err, style, lw=2, label=label)

            ax_mean.grid(True)
            ax_error.grid(True)
            ax_mean.legend()
            ax_error.legend()

        plt.tight_layout()
        plt.show()

A = 1 
f = 300 #Hzc
phi = 0

test_signals = frameworks.Generate_Signals(A,f,phi)
noisy_test_signals = frameworks.Noisy_Signals(test_signals)

print(len(test_signals.pure_signal))
print(len(test_signals.note))
print(len(test_signals.vocal))
#
test(noisy_test_signals.pure_signal, noisy_test_signals.fs, "Simple Sine Wave")
#test(noisy_test_signals.descending_harmonics, noisy_test_signals.fs, "Harmonic Sine Wave Decending")
#test(noisy_test_signals.shuffled_harmonic, noisy_test_signals.fs, "Harmonic Sine Wave Shuffeled")
#test(noisy_test_signals.abrupt_change, noisy_test_signals.fs, "Abrupt Change")
#test(noisy_test_signals.note, noisy_test_signals.note_fs, "B-oboe Note", len(test_signals.note), 6)
#test(noisy_test_signals.vocal, noisy_test_signals.vocal_fs, "Zauberflote Vocal", len(test_signals.vocal), 6)
#