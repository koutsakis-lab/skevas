# -*- coding: utf-8 -*-
import numpy as np
from scipy import signal
from datetime import datetime
import matplotlib.pyplot as plt

now = datetime.now() # current date integer in YYYYMMDD format
todaySeed=int(now.strftime('%Y%m%d'))

def makeNoise(σNoise, signal):
    
    if σNoise==0:
        return 0, float('NaN') # In absence of noise, the SNR is not defined.
    
    # Time-series signal of noise
    N = np.size(signal)
    rng = np.random.default_rng(seed=todaySeed) # construct a random number generator
    noise = rng.normal(scale=σNoise, size=N) # [K or C] instantaneous noise
    
    # Signal-to-Noise Ratio
    strengthNoise = 4*σNoise # [K or C] (σ=ptp/4) Standard deviation (rough estimate) of the measurement error
    strengthSignal = np.ptp(signal) # peak-to-peak / max - min
    SNR_power = strengthSignal/strengthNoise
    
    return noise, SNR_power

def RootMeanSquaredDifference(actual, estimate):
    return np.sqrt(np.mean((actual-estimate)**2))

def guaranteedConvergence(x, sz, relTol):
    last = x[-sz:] # last cycle
    before_last = x[-2*sz:-sz] # second to last cycle
    relError = np.sqrt(np.mean(((last-before_last))**2)) / np.mean(before_last) # get relative error between the last two cycles normalized by the second to last cycle
    return relError < relTol

def butterworthLowPassFilter(data, fCutoff, fSampling, order=4):
    fNyquist = 0.5 * fSampling
    fCutoffNormal = float(fCutoff) / fNyquist
    b, a = signal.butter(order, fCutoffNormal, btype='lowpass')
    return signal.filtfilt(b, a, data)

def powerSpectralDensity(signal_time, fSampling):
    f_psd, signal_psd = signal.periodogram(signal_time, fSampling)
    
    if f_psd[0] == 0: # remove the zero frequency from the PSD data
        f_psd=f_psd[1:]
        signal_psd=signal_psd[1:]

    return f_psd, signal_psd




# powerSignalInstantaneous = np.abs((signal-np.mean(signal))**2) # Substract the steady part from the signal to avoid effects on the RMS calculations below
# powerSignal = np.mean(powerSignalInstantaneous) # Average signal power in the time domain
# RMS_signal = np.sqrt(powerSignal) # RMS of signal

# RMS_noise = np.sqrt(RMS_signal**2 / SNR_power)
# powerNoise = powerSignal*(RMS_noise/RMS_signal)**2

    # Calculate instantaneous noise

 # The standard deviation is given by: σ = np.sqrt(1/(N-1)*Σ(xi-μ)**2)
 # The term N−1 corresponds to the number of degrees of freedom in the vector of deviations from the mean (x0-x_bar...xn-x_bar)
 # In this case, zero mean is assumed for the noise, μ=0, therefore σ can be solved using average power P=1/N*Σxi**2
 # => Σxi**2=N*P which gives

 # σ = np.sqrt(N/(N-1)*powerNoise) # [K or C] corrected sample standard deviation of the measurement error