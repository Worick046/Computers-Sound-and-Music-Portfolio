#Some parts of this program are taken from https://stackoverflow.com/questions/25191620/creating-lowpass-filter-in-scipy-understanding-methods-and-units
#This allowed for understanding how to create a lowpass filter in python.


from random import sample
import numpy as np
import scipy.signal as signal
import sounddevice as sd
from scipy.io import wavfile
import matplotlib.pyplot as plt


lowCutoff = 300
midCutoff = [300, 2000]
highCutoff = 2000
sampleRate = 48000
lowNormCutoff = lowCutoff / (sampleRate / 2)
midNormCutoff = [midCutoff[0] / (sampleRate / 2), midCutoff[1] / (sampleRate / 2)]
highNormCutoff = highCutoff / (sampleRate / 2)
lowcoeffs = signal.butter(5, lowNormCutoff, "low")
midcoeffs = signal.butter(5, midNormCutoff, "bandpass")
highcoeffs = signal.butter(5, highNormCutoff, "high")

lowfilterdelay = signal.lfilter_zi(lowcoeffs[0], lowcoeffs[1])
midfilterdelay = signal.lfilter_zi(midcoeffs[0], midcoeffs[1])
highfilterdelay = signal.lfilter_zi(highcoeffs[0], highcoeffs[1])


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

def lowPassFilter(wave, sampleRate, strength):
    global lowcoeffs
    global lowfilterdelay
    filteredWave, lowfilterdelay = signal.lfilter(lowcoeffs[0], lowcoeffs[1], wave, zi=lowfilterdelay)
    return filteredWave

def highPassFilter(wave, sampleRate, strength):
    global highcoeffs
    global highfilterdelay
    filteredWave, highfilterdelay = signal.lfilter(highcoeffs[0], highcoeffs[1], wave, zi=highfilterdelay)
    return filteredWave

def bandPassFilter(wave, sampleRate, strength):
    global midcoeffs
    global midfilterdelay
    filteredWave, midfilterdelay = signal.lfilter(midcoeffs[0], midcoeffs[1], wave, zi=midfilterdelay)
    return filteredWave


def getBandEnergyArray(data, sampleRate, window):
    shift = len(window)
    numberOfSlices = len(data) // shift
    bandEnergyArray = []
    for i in range(numberOfSlices):
        bandEnergyArray.append(getFrequencyBandEnergies(data, window, shift * i, sampleRate))

    return bandEnergyArray



def applyToneControl(data, sampleRate, bandEnergyArray, window):
    shift = len(window)
    filterData = np.zeros(len(data))
    for i in range(len(bandEnergyArray)):
        #Separate Bands
        lowData = lowPassFilter(data[shift * i:shift * (i + 1)], 48000, 5)
        midData = bandPassFilter(data[shift * i:shift * (i + 1)], 48000, 5)
        highData = highPassFilter(data[shift * i:shift * (i + 1)], 48000, 5)

        average = np.average(np.array(bandEnergyArray[i]))
        filteredBandEnergyArray = getFrequencyBandEnergies(lowData + midData + highData, window, 0, sampleRate)
        #Scale each band to average
        lowscale = average / filteredBandEnergyArray[0]
        midscale = average / filteredBandEnergyArray[1]
        highscale = average / filteredBandEnergyArray[2]
        lowData = np.multiply(lowData, lowscale)
        midData = np.multiply(midData, midscale)
        highData = np.multiply(highData, highscale)


        #Combine the bands back together
        filterData[shift * i: shift * (i + 1)] = lowData + midData + highData
    return filterData
        



def plotFrequencies(wave, window):
    x = np.linspace(0, len(window), len(window), dtype=np.int32)
    plt.plot(x, wave[:len(window)] * window)
    plt.show()


if __name__ == "__main__":
    data = generateSineWaves(48000, [220, 440, 660, 6000], [0.1, 0.1, 0.1, 0.1], 3)
    data2 = generateSineWaves(48000, [220, 440, 660, 6000], [0.2, 0.3, 0.3, 0.2], 3)
    data = np.concatenate((data, data2))
    lowfilteredData = lowPassFilter(data, 48000, 5)
    midfilteredData = bandPassFilter(data, 48000, 5)
    highfilteredData = highPassFilter(data, 48000, 5)
    window = np.hanning(48000 // 10)
    bandEnergyArray = getBandEnergyArray(data, 48000, window)
    filteredData = applyToneControl(data, 48000, bandEnergyArray, window)
    filterBandEnergyArray = getBandEnergyArray(filteredData, 48000, window)
    for i in range(len(bandEnergyArray)):
        print(bandEnergyArray[i], filterBandEnergyArray[i])

    wavfile.write("Test.wav", 48000, filteredData)
