#Some parts of this program are taken from https://stackoverflow.com/questions/25191620/creating-lowpass-filter-in-scipy-understanding-methods-and-units
#This allowed for understanding how to create a lowpass filter in python.


import numpy as np
import scipy.signal as signal
import sounddevice as sd
from scipy.io import wavfile
import matplotlib.pyplot as plt

def generateSineWave(sampleRate, frequency, amplitude, duration):
    wave = np.linspace(0, 2 * np.pi * frequency * duration, sampleRate * duration)
    wave = amplitude * np.sin(wave)
    return wave

def generateSineWaves(sampleRate, frequencies, amplitudes, duration):
    if len(frequencies) != len(amplitudes):
        print("Error: The amount of frequencies", len(frequencies), "does not match the amount of amplitudes", len(amplitudes))
        return

    wave = generateSineWave(sampleRate, frequencies[0], amplitudes[0], duration)
    for i in range(1, len(frequencies)):
        wave = np.add(wave, generateSineWave(sampleRate, frequencies[i], amplitudes[i], duration))
    return wave

def integer_format(wave):
    intwave = np.zeros(len(wave), dtype=np.int16)
    tempwave = np.multiply(wave, 32767)
    intwave = np.add(intwave, tempwave.astype(np.int16))
    return intwave

def getFrequencyBandEnergies(wave, window):
    fft = np.fft.rfft(wave[:len(window)] * window)
    #fft = np.fft.rfft(wave)
    amplitudes = np.abs(fft)
    #print(amplitudes.argmax() // 2, amplitudes.max())
    lowband = 0
    midband = 0
    highband = 0
    maximum = amplitudes.max()
    for i in range(len(amplitudes) // 2):
        #if(amplitudes[i] < maximum / 16):
        #    amplitudes[i] = 0
        #print(i // 2, amplitudes[i])
        if i // 2 <= 300:
            lowband += amplitudes[i]
        if i // 2 > 300 and i // 2 <= 2000:
            midband += amplitudes[i]
        if i // 2 > 2000:
            highband += amplitudes[i]
    return [lowband, midband, highband]

def lowpassFilter(wave, sampleRate):
    cutoff = 300.0
    normalized_cutoff = cutoff / (sampleRate / 2)
    b, a = signal.butter(5, normalized_cutoff, btype='low')
    filteredWave = signal.lfilter(b, a, wave)
    return filteredWave

def plotFrequencies(wave, window):
    x = np.linspace(0, len(window), len(window), dtype=np.int32)
    plt.plot(x, wave[:len(window)] * window)
    plt.show()


if __name__ == "__main__":
    data = generateSineWaves(48000, [220, 440, 660, 6000], [0.1, 0.1, 0.01, 0.1], 2)
    filteredData = lowpassFilter(data, 48000)
    window = np.hanning(48000 * 2)
    print(getFrequencyBandEnergies(data, window))
