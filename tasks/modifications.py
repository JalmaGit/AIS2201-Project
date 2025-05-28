from scipy.fft import fft, ifft, fftfreq, rfft
import numpy as np

class detection:

    def __init__(self):
        pass


    def simple_detection(self, xn, fs):
        
        N = len(xn)
        Xm = np.abs(np.fft.rfft(xn))
        Xm[0] = 0
        estimated_frequency = np.argmax(Xm) * fs/N

        return estimated_frequency
    
    def auto_correlation_pitch(self, xn, fs, fmin=20, fmax=4200):
        Xm = np.fft.fft(xn)
        Xm[0] = 0
        x = np.fft.ifft(Xm).real

        N = 2*len(x)
        Xm = np.fft.fft(x, N)
        Rx = np.abs(Xm)**2
        R = np.fft.ifft(Rx).real
        R = R[:len(x)]        
        R /= R[0]         

        minlag = int(fs / fmax)
        maxlag = int(fs / fmin)

        lag = None
        maxlag_clamped = min(maxlag, len(R)-1)
        segment = R[minlag:maxlag_clamped]
        lag = np.argmax(segment) + minlag

        return fs / lag
        
        
    def harmonic_spectrum(self, xn, num_harmonics, mode, fs, p=2):
        Xm = np.fft.fft(xn)
        Xm_abs = np.abs(Xm[:len(Xm)//2]) 
        Xm_abs[0] = 0
        N = len(Xm_abs)

        hps_result = []

        for k in range(1, N // num_harmonics): 
            product = Xm_abs[k]
            for r in range(2, num_harmonics + 1):
                harmonic_index = k * r
                if harmonic_index < N:
                    if mode == "product":
                        product *= Xm_abs[harmonic_index]
                    elif mode == "sum":
                        product += Xm_abs[harmonic_index]
                    elif mode == "pow":
                        product += Xm_abs[harmonic_index] ** p
                else:
                    break
            hps_result.append(product)

        estimated_fundamental_bin = hps_result.index(max(hps_result)) + 1
        fundamental_freq = estimated_fundamental_bin * fs / len(xn)
        
        return fundamental_freq

class Tools:

    def __init__(self):
        pass

    def chebyshev_window(y, N):
        M = N-1
        m = np.arange(M)
        alpha = np.cosh(1/M*np.arccosh(10**y))
        Am = np.abs(alpha * np.cos(np.pi*m/M))

        Wm = np.zeros(M)
        for index, element in enumerate(Am):
            if element > 1:
                Wm[index] = (-1)**index*np.cosh(M*np.arccosh(np.abs(element)))
            elif element <= 1:
                Wm[index] = (-1)**index*np.cos(M*np.arccos(element))
        
        wn = np.fft.ifft(Wm).real
        wn[0] = wn[0]/2
        wn = np.append(wn,wn[0])
        wn = wn/np.max(wn)

        return wn
    
    def hamming_window(N):
        n = np.arange(N)
        wn = 0.54 - 0.46*np.cos(2*np.pi*n/N)
        return wn

    def hanning_window(N):
        n = np.arange(N)
        wn = 0.5 - 0.5*np.cos(2*np.pi*n/N)
        return wn

    def blackman_window(N):
        n = np.arange(N)
        wn = 0.42-0.5*np.cos(2*np.pi*n/(N-1))+0.08*np.cos(4*np.pi*n/(N-1))
        return wn
    
    def boxcar_window(N):
        wn = np.ones(N)
        return wn

    def _factorial(n):
        result = 1
        for i in range(2, n + 1):
            result *= i
        return result   


    def _zero_order_bezzel(x):
        I_o = np.zeros(len(x))
        for q in range(24):
            I_o = I_o + (x**(2*q))/((4**q)*Tools._factorial(q)**2)
        return I_o


    def kaiser_window(N, beta):
        n = np.arange(N)
        p = (N-1)/2

        top = Tools._zero_order_bezzel((beta*np.sqrt(1-((n-p)/p)**2)))
        bottom = Tools._zero_order_bezzel(np.array([beta]))
        
        wn = top/bottom[0]

        return wn
    
    def fast_convolution(xn, hn):
        L = len(xn) + len(hn) - 1

        xn_padded = np.pad(xn, (0, L - len(xn)))
        hn_padded = np.pad(hn, (0, L - len(hn)))

        Xm = np.fft.fft(xn_padded)
        Hm = np.fft.fft(hn_padded)
        Ym = Xm * Hm

        yn = np.fft.ifft(Ym).real

        return yn



        
class preprocessing:

    def __init__(self, buffer_size = 2048, y = 2.5, beta = 4):
        self.y = y
        self.beta = beta
        self.prev_buffer = np.zeros(buffer_size)
        self.buffer_size = buffer_size

    def sliding_buffer(self, xn, buffer_overlap, fs):
        new_size = len(self.prev_buffer) - buffer_overlap
        self.prev_buffer = np.concatenate((self.prev_buffer[new_size:], xn))
        return self.prev_buffer

    def zero_padding(self, xn, desired_resolution, fs):
        N = len(xn)
        L = fs/desired_resolution
        Z = L - N
        yn = np.pad(xn, (0,int(Z)), "constant")

        return yn
    
    def moving_average_window(self, xn, N, window, fs):
        hn = np.ones(N)/N

        wn = hn.copy()
        if window == "hanning":
            wn = hn * Tools.hanning_window(N)
        elif window == "hamming":
            wn = hn * Tools.hamming_window(N)
        elif window == "blackman":
            wn = hn * Tools.blackman_window(N)
        elif window == "chebyshev":
            wn = hn * Tools.chebyshev_window(self.y, N)
        elif window == "kaiser":
            wn = hn * Tools.kaiser_window(self.beta, N)
        elif window == "boxcar":
            wn = hn * Tools.boxcar_window(N)

        yn = Tools.fast_convolution(xn, wn)
        
        return yn
    
    def band_pass_filter(self, xn, low_cut, high_cut, taps, window, fs, A=2):
        N = taps
        D = N//2 # 
        w_shift = (((low_cut + high_cut)/2)*2*np.pi)/fs
        wc =  (((high_cut - low_cut)/2)*2*np.pi)/fs

        n = np.arange(N)
        hn = wc/np.pi * np.sinc(wc/np.pi*(n))

        Hw = np.fft.fft(hn)
        freq = np.fft.fftfreq(N) * 2 * np.pi

        Hd = np.exp(-1j*freq*D) * Hw.copy()
        hd = np.fft.ifft(Hd).real

        hd[:N//2] = np.flip(hd[N//2+1:])

        if window == "hanning":
            hd = hd * Tools.hanning_window(N)
        elif window == "hamming":
            hd = hd * Tools.hamming_window(N)
        elif window == "blackman":
            hd = hd * Tools.blackman_window(N)
        elif window == "chebyshev":
            hd = hd * Tools.chebyshev_window(self.y, N)
        elif window == "kaiser":
            hd = hd * Tools.kaiser_window(self.beta, N)
        elif window == "boxcar":
            hd = hd * Tools.boxcar_window(N)
        
        hn_bp = hd.copy()*A*np.cos(w_shift*(n-D))
        
        yn = Tools.fast_convolution(xn,hn_bp)

        return yn