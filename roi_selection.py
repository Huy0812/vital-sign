from cmath import isnan, nan
import numpy as np
import scipy as sp
import heartpy as hp
def bandpass_filter(signal, time) :
    signal = np.array(signal)
    fft = np.fft.rfft(signal) # FFT
    #fft_data = (np.abs(fft_data))**2
    #ffty=np.fft.fft(signal)
    #power spectrum, after real2complex transfrom (factor )
    #scale=2.0/(len(signal))
    #ffty= scale*abs2(ffty)
    #ffty = np.array(ffty)
    
    freq=abs(np.fft.rfftfreq(len(signal) , 1./(len(signal)/time)))
    power = abs((fft**2) * freq) 
    #power = abs(fft)
    #inds= np.where((freq < 0.7) | (freq > 4) )
    #power[inds] = 0
    #signal = np.fft.ifft(fft_data)

    # power spectrum, via scipy welch. 'boxcar' means no window, nperseg=len(y) so that fft computed on the whole signal.
    #freq = abs(np.fft.fftfreq(len(signal), 1./(len(signal)/time))) # Frequency data
    #ft_data = fft_data*freq
    #wd, m = hp.process(signal, len(signal)/time)
    #print(m)
    #freq,power=sp.signal.welch(signal, fs=len(signal)/time,nperseg=len(signal))
    inds= np.where((freq < 0.7) | (freq > 4) )
    power[inds] = 0
    #power = wd['psd']
    #frq = wd['frq']
    #print(power, frq)
    #if np.isnan(wd['psd']).any() :
        #power = 0
    #if np.isnan(wd['frq']).any() :
        #frq = 0
    
    #peaks,_ = sp.signal.find_peaks(power)
    return power, freq
    #bps_freq=60.0*freq
    #max_index = np.argmax(fft_data)
    #HR =  bps_freq[max_index]
def selection_signal(fft_forehead, bpm_forehead, fft_nose, bpm_nose, fft_face, bpm_face) :
    max_index_forehead  = np.argmax(fft_forehead)
    max_index_nose  = np.argmax(fft_nose)
    max_index_face  = np.argmax(fft_face)
    max_psd = max(np.max(fft_forehead), np.max(fft_nose), np.max(fft_face))
    print(60*bpm_forehead[max_index_forehead])
    print(60*bpm_nose[max_index_nose])
    print(60*bpm_face[max_index_face])

    if max_psd == np.max(fft_forehead) :
        print("forehead")
        return fft_forehead, bpm_forehead, max_index_forehead
    if max_psd == np.max(fft_nose) :
        print("nose")
        return fft_nose, bpm_nose, max_index_nose
    if max_psd == np.max(fft_face) :
        print("face")
        return fft_face, bpm_face, max_index_face


    
    
