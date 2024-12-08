import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from IPython.display import Audio

class detection_algorithms:

    def __init__(self):
        self.fs = 0

        
    def set_new_fs(self, fs):
        self.fs = fs

    def simple_detection(self, fs, sample):
        N = len(sample)
        Xm = sp.fft.rfft(sample).__abs__()
        f_est = np.argmax(Xm)*fs/N

        return f_est
    
    def harmonic_product_spectrum(self, fs, sample, harmonics):
        N = len(sample)
        Xm = np.abs(np.fft.rfft(sample))
        hps = Xm.copy()
        for h in range(2, harmonics):  # Number of harmonics
            downsampled = Xm[::h]
            hps[:len(downsampled)] *= downsampled
        f_est = np.argmax(hps) * fs / N

        return f_est
        
    def frequency_domain_autocorrelation(self,fs, sample):
    # Compute the Fourier transform of the input signal
        fft_signal = np.fft.fft(sample)
        
        # Multiply the Fourier transform by its conjugate
        power_spectrum = fft_signal * np.conjugate(fft_signal)
        
        # Compute the inverse Fourier transform of the product
        autocorrelation = np.fft.ifft(power_spectrum)

        # Find the first peak in the autocorrelation
        for lag in range(1, len(autocorrelation)):
            if autocorrelation[lag] > autocorrelation[lag - 1] and autocorrelation[lag] > autocorrelation[lag + 1]:
                fundamental_frequency = fs / lag
                return fundamental_frequency
        return 0
    
    def spectral_subtraction(self, signal, noise, fs):
        # Estimate the noise spectrum
        noise_spectrum = np.abs(np.fft.fft(noise))
        
        # FFT of the signal
        signal_spectrum = np.fft.fft(signal)
        
        # Subtract noise spectrum
        clean_spectrum = np.abs(signal_spectrum) - noise_spectrum
        clean_spectrum = np.maximum(clean_spectrum, 0)  # Avoid negative values
        
        # Reconstruct the signal
        clean_signal = np.fft.ifft(clean_spectrum * np.exp(1j * np.angle(signal_spectrum))).real
        return clean_signal


    def increase_spectral_resolution(self, x_n, padding):
        pad_factor = padding
        x_n_padded = np.concatenate((x_n,np.zeros(int(len(x_n)*pad_factor))))

        return x_n_padded


    def low_pass(self, fs, x_n,):
        f_upper = 4200
        f_lower = 25
        f_c=(f_upper-f_lower)/2         #Cuttoff Frequency
        M = M_min=1337        #M_min

        ## I need some Propper explenation  

        D = M//2 #This is also the Counterpoint
            
        n = np.arange(0,M) 
        omega_c = f_c*2*np.pi/fs
        hn = omega_c/np.pi*np.sinc((omega_c/np.pi)*n)
            
        h_D = np.zeros(M)
            
        for element in n:
            h_D[element] = hn[(element-D)]
                
        h_D[0:D] = np.flip(h_D[D+1:M])

        f_center = (f_upper + f_lower)/2  # Submit in radians
        omega_center =2*np.pi*f_center/fs

        hn_bp = 2*h_D*np.cos(omega_center*n)   
        
        w = 0.54 - 0.46 * np.cos((2 * np.pi * n) / (M - 1))
            
        h_D4 = hn_bp.copy() * w

        filtered_sig = np.convolve(x_n, h_D4, mode='same')
        """   
        Xmo = np.fft.rfft(x_n).__abs__()
        Xm = np.fft.rfft(filtered_sig).__abs__()
        frequencies = np.fft.rfftfreq(len(x_n), 1 / fs)
        plt.plot(frequencies, Xmo)
        plt.plot(frequencies, Xm)
        plt.show()

        Hm = np.fft.fft(h_D4)        
        frequencies = np.fft.fftfreq(len(h_D4)) * 2 
        Hd4_mag = 20*np.log10(np.abs(Hm[:len(h_D4)//2]))
        
        plt.close("all")
        fig, ax = plt.subplots(figsize=(12,7))
        ax.plot(frequencies[:len(h_D4)//2],Hd4_mag, label="Low Pass Filter")
        ax.legend(loc="best")
        plt.show()
        """
        return filtered_sig
