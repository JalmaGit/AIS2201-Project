import numpy as np
import scipy.io.wavfile as wavfile

class frameworks:
    
    def __init__(self, A, f, phi):

        self.fs = 48000 # Sampling Frequency
        duration = 5 # Seconds
        n = np.linspace(0,5,self.fs*duration) 

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
        self.note_fs, self.note = self.load_note("data/B_oboe.wav")

        #Sixth Test Signal
        self.vocal_fs, self.vocal = self.load_vocal("data/Zauberflöte_vocal.wav")


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
        second_half = self.generate_pure_signal(A,f*1/2,n_1,phi)

        return np.concatenate((first_half, second_half))
    
    def load_note(path):
        fs, sampleData = wavfile.read(path) 
        xn_norm = sampleData / np.max(np.abs(yn))
        return fs, xn_norm
    
    def load_vocal(path):
        fs, sampleData = wavfile.read(path)
        xn_norm = sampleData / np.max(np.abs(yn))
        return fs, xn_norm 
        



    


    
