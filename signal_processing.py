from cmath import sin
from locale import normalize
from unittest import result
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy import sparse
from sklearn.decomposition import FastICA


class Signal():
    def __init__(self):
        self.signal = []
        self.length = 0

    def __call__(self, fps):
        self.eliminate_motion(fps)
        self.denoise_filter(self.signal)
        self.normalization()
        #self.ICA()
        self.detrending_filter()
        self.moving_avg_filter()
        pass

    def eliminate_motion(self, time):
        if (self.SNR(time) < 0):
            signal_arr = np.array_split(self.signal, 10)
            std_max = 0
            std_index = 0
            for i in range(0, 9):
                if signal_arr[i].std() > std_max:
                    std_max = signal_arr[i].std()
                    std_index = i
            count = 1
            temp = len(signal_arr[std_index])
            for i in range(0, len(signal_arr[std_index])):
                if count <= temp * 0.05:
                    signal_arr[std_index] = np.delete(signal_arr[std_index], np.argmax(signal_arr[std_index]))
                    count += 1
            k = 0
            result_signal = np.concatenate(signal_arr[:k] + signal_arr[k + 1:], axis=0)

            self.signal = result_signal

    def SNR(self, time):
        signal = np.array(self.signal)
        # self.length = len(signal)
        fft = np.fft.fft(signal)
        freq = abs(np.fft.fftfreq(len(fft), 1. / (len(fft) / time)))
        power = fft
        #power = signal
        # freq,power=sp.signal.welch(signal, fs=len(signal)/time,window='boxcar',nperseg=len(signal),scaling='spectrum', axis=-1, average='mean')
        inds = np.where((freq < 0.5) | (freq > 4))
        power[inds] = 0
        power = (power - min(power)) / (max(power) - min(power))
        max_index = np.argmax(power)
        sum_signal = (sum(power[max_index:max_index + 4]))
        sum_noise = abs(4 - sum_signal)
        snr = 20 * np.log10(sum_signal / sum_noise)
        #print(snr)
        return snr

    def denoise_filter(self, signal, threshold=1):
        arr = []
        arr.append(signal[0])
        i = 0
        while i < len(signal) - 1:
            temp = signal[i + 1] - signal[i]
            if abs(temp) > threshold:
                for j in range(i + 1, len(signal)):
                    signal[j] = signal[j] - temp
            else:
                arr.append(signal[i + 1])
                i = i + 1
        self.signal = arr

    def normalization(self):
        mean = np.array(self.signal).mean()
        std = np.array(self.signal).std()
        self.signal = (self.signal - mean) / std

    def moving_avg_filter(self, w_s=5):
        ones = np.ones(w_s) / w_s
        self.signal = np.convolve(self.signal, ones, 'valid')

    def detrending_filter(self, regularization=10):  # smoothing parameter λ = 10
        N = len(self.signal)
        identity = np.eye(N)
        B = np.dot(np.ones((N - 2, 1)), np.array([[1, -2, 1]]))
        D_2 = sp.sparse.dia_matrix((B.T, [0, 1, 2]), shape=(N - 2, N)).toarray()
        inv = np.linalg.inv(identity + regularization ** 2 * D_2.T @ D_2)
        z_stat = (identity - inv) @ self.signal
        trend = np.squeeze(np.asarray(self.signal - z_stat))
        self.signal = np.array(self.signal) - trend

    def ICA(self):
        ica = FastICA()
        self.signal = np.reshape(self.signal, (len(self.signal) , 1))
        self.signal = ica.fit_transform(self.signal)
        self.signal = self.signal.flatten()
