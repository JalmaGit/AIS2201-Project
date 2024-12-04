from detection import detection_algorithms as da
from signal_generation import signal_generation as sg


fs = 48000
signals = sg()

detect = da(fs)