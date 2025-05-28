import numpy as np
import scipy.io.wavfile as wavfile
import os

class Generate_Signals:
    
    def __init__(self, A, f, phi):

        self.A = A
        self.noise_var = 1
        self.fs = 48000 #Sampling Frequency
        self.duration = 4 #Seconds
        n = np.linspace(0,self.duration,self.fs*self.duration) 

        #First Test Signal
        self.pure_signal = self.generate_pure_signal(A, f, n, phi) 
        
        #Second Test Signal
        descending_amplitudes = np.array([A,A*1/2,A*1/4,A*1/8]) 
        phases = np.array([phi,phi,phi,phi]) 
        self.descending_harmonics = self.generate_hamonic(descending_amplitudes, f, n, phases)

        #Third Test Signal
        shuffled_amplitudes = np.array([A*1/4,A*1/2,A,A*1/8]) 
        phases = np.array([phi,phi,phi,phi]) 
        self.shuffled_harmonic = self.generate_hamonic(shuffled_amplitudes, f, n, phases) 
 
        #Fourth Test Signal
        self.abrubt_change = self.generate_abrupt_change(A,f,n,phi) 

        #Fifth Test Signal
        self.note_fs, self.note = self.load_note("tasks/data/B_oboe.wav") 

        #Sixth Test Signal
        self.vocal_fs, self.vocal = self.load_vocal("tasks/data/Zauberflöte_vocal.wav") 

    def generate_pure_signal(self, A, f, n, phi):
        return A * np.cos(2*np.pi*f*n + phi)
    
    def generate_hamonic(self, A, f, n, phi):

        first_harmonic = self.generate_pure_signal(A[0], f, n, phi[0])
        second_harmonic = self.generate_pure_signal(A[1], f*2, n, phi[1])
        third_harmonic = self.generate_pure_signal(A[2], f*3, n, phi[2])
        fourth_harmonic = self.generate_pure_signal(A[3],f*4, n, phi[3])

        return first_harmonic + second_harmonic + third_harmonic + fourth_harmonic
    
    def generate_abrupt_change(self, A, f, n, phi):
        n_0, n_1 = np.split(n,2) 
    
        first_half = self.generate_pure_signal(A,f,n_0,phi)
        second_half = self.generate_pure_signal(A,f*3/2,n_1,phi)

        return np.concatenate((first_half, second_half))
    
    def load_note(self, path):        
        fs, sampleData = wavfile.read(path) 
        xn_norm = sampleData / np.max(np.abs(sampleData))
        return fs, xn_norm
    
    def load_vocal(self, path):
        fs, sampleData = wavfile.read(path)
        xn_norm = sampleData / np.max(np.abs(sampleData))


        return fs, xn_norm 


class Noisy_Signals:

    def __init__(self, signals: Generate_Signals):
        
        self.A = signals.A
        self.fs = signals.fs
        self.pure_signal = self.add_noise(signals.pure_signal)
        self.descending_harmonics = self.add_noise(signals.descending_harmonics)
        self.shuffled_harmonic = self.add_noise(signals.shuffled_harmonic)
        self.abrupt_change = self.add_noise(signals.abrubt_change)
        self.note_fs = signals.note_fs
        self.note = self.add_noise(signals.note, 5)
        self.vocal_fs = signals.vocal_fs
        self.vocal = self.add_noise(signals.vocal,5)
        
    def add_noise(self, xn, loops=10):

        noisy_signal = []
        noisy_signal.append(xn.copy())

        SNR = np.logspace(-2,4, num=loops)
        noise_vars = ((self.A**2)/2)/(1/SNR)

        for i in range(loops):
            noise = np.random.normal(scale=np.sqrt(noise_vars[i]), size=len(xn))  # Gaussian noise
            noisy_signal.append(xn.copy() + noise.copy())

        yn = np.concatenate(noisy_signal)

        yn_norm = yn / np.max(np.abs(yn))    

        return yn_norm
