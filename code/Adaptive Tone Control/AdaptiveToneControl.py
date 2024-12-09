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

def getFrequencyBandEnergies(wave, window, shift, sampleRate):
    fft = np.fft.rfft(wave[shift:len(window) + shift] * window)
    #fft = np.fft.rfft(wave)
    amplitudes = np.abs(fft)
    frequencies = np.fft.rfftfreq(len(window), d=1./sampleRate)
    #print(amplitudes.argmax() // 2, amplitudes.max())

    #Filter amplitudes through a threshold
    amplitudes[amplitudes < 100.] = 0.0

    #Filter amplitudes into three frequency categories 0 - 300, 300 - 2000, 2000+ . Any amplitudes that are at 0 are discarded.
    lowband = []
    midband = []
    highband = []
    for i in range(len(amplitudes)):
        if frequencies[i] <= 300 and amplitudes[i] > 0:
            lowband.append(amplitudes[i])
        elif frequencies[i] > 300 and frequencies[i] <= 2000 and amplitudes[i] > 0:
            midband.append(amplitudes[i])
        elif frequencies[i] > 2000 and amplitudes[i] > 0:
            highband.append(amplitudes[i])


    #Take the average amplitude at each of the categories or set the average to 0 if there are no amplitudes in the band.
    if len(lowband) > 0:
        lowband = np.average(np.array(lowband))
    else:
        lowband = 0

    if len(midband) > 0:
        midband = np.average(np.array(midband))
    else:
        midband = 0

    if len(highband) > 0:
        highband = np.average(np.array(highband))
    else:
        highband = 0

    rescale = sampleRate / len(window)
    return [lowband * rescale, midband * rescale, highband * rescale]

def lowpassFilter(wave, sampleRate, strength):
    cutoff = 300.0
    normalized_cutoff = cutoff / (sampleRate / 2)
    b, a = signal.butter(strength, normalized_cutoff, btype='low')
    filteredWave = signal.lfilter(b, a, wave)
    return filteredWave

def highpassFilter(wave, sampleRate, strength):
    cutoff = 2000.0
    normalized_cutoff = cutoff / (sampleRate / 2)
    b, a = signal.butter(strength, normalized_cutoff, btype='high')
    filteredWave = signal.lfilter(b, a, wave)
    return filteredWave

def bandPassFilter(wave, sampleRate, strength):
    cutoffs = [300.0, 2000.0]
    normalized_cutoffs = [cutoffs[0] / (sampleRate / 2), cutoffs[1] / (sampleRate / 2)]
    b, a = signal.butter(strength, normalized_cutoffs, btype='bandpass')
    filteredWave = signal.lfilter(b, a, wave)
    return filteredWave


def getBandEnergyArray(data, sampleRate, window):
    numberOfSamples = len(data)
    shift = sampleRate // 10
    numberOfFullShifts = numberOfSamples // shift - len(window) // shift + 1
    bandEnergyArray = []
    for i in range(numberOfFullShifts):
        bandEnergyArray.append(getFrequencyBandEnergies(data, window, shift * i))
    return bandEnergyArray

def applyToneControl(data, sampleRate, bandEnergyArray):
    shift = sampleRate // 10
    filterData = np.zeros(len(data))
    for i in range(len(bandEnergyArray)):
        print()
    return filterData
        



def plotFrequencies(wave, window):
    x = np.linspace(0, len(window), len(window), dtype=np.int32)
    plt.plot(x, wave[:len(window)] * window)
    plt.show()


if __name__ == "__main__":
    data = generateSineWaves(48000, [220, 440, 660, 6000], [0.1, 0.1, 0.0, 0.1], 3)
    data2 = generateSineWaves(48000, [220, 440, 660, 6000], [0.2, 0.1, 0.3, 0.2], 3)
    data = np.concatenate((data, data2))
    lowfilteredData = lowpassFilter(data, 48000, 5)
    midfilteredData = bandPassFilter(data, 48000, 5)
    highfilteredData = highpassFilter(data, 48000, 5)
    window = np.hanning(48000 // 10)
    print(len(lowfilteredData))
    getFrequencyBandEnergies(lowfilteredData, window, 0, 48000)
    print(getFrequencyBandEnergies(data, window, 0, 48000))
    print(getFrequencyBandEnergies(lowfilteredData, window, 0, 48000))
    print(getFrequencyBandEnergies(midfilteredData, window, 0, 48000))
    print(getFrequencyBandEnergies(highfilteredData, window, 0, 48000))
