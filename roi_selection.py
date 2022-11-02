from cmath import isnan, nan
import numpy as np
import scipy as sp
import heartpy as hp


def bandpass_filter(signal, time):
    signal = np.array(signal)
    fft = np.fft.rfft(signal)  # FFT
    freq = abs(np.fft.rfftfreq(len(signal), 1. / (len(signal) / time)))
    power = abs((fft ** 2) * freq)
    inds = np.where((freq < 0.7) | (freq > 4))
    power[inds] = 0
    return power, freq


def selection_signal(fft_forehead, bpm_forehead, fft_nose, bpm_nose, fft_face, bpm_face):
    max_index_forehead = np.argmax(fft_forehead)
    max_index_nose = np.argmax(fft_nose)
    max_index_face = np.argmax(fft_face)
    max_psd = max(np.max(fft_forehead), np.max(fft_nose), np.max(fft_face))
    #max_psd = np.max(fft_nose)
    #print(60 * bpm_forehead[max_index_forehead])
    #print(60 * bpm_nose[max_index_nose])
    #print(60 * bpm_face[max_index_face])

    if max_psd == np.max(fft_forehead):
        print("forehead")
        return fft_forehead, bpm_forehead, max_index_forehead

    if max_psd == np.max(fft_nose):
        print("nose")
        return fft_nose, bpm_nose, max_index_nose

    if max_psd == np.max(fft_face):
        print("face")
        return fft_face, bpm_face, max_index_face
