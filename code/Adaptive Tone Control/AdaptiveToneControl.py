#Some parts of this program are taken from https://stackoverflow.com/questions/25191620/creating-lowpass-filter-in-scipy-understanding-methods-and-units
#This allowed for understanding how to create a lowpass filter in python.


from random import sample
import numpy as np
import scipy.signal as signal
import sounddevice as sd
from scipy.io import wavfile
import matplotlib.pyplot as plt
import sys

#SampleRate of the audio file
sampleRate = None

#Filter state variables
lowcoeffs = None
midcoeffs = None
highcoeffs = None

#Filter variables required for smooth filtering over multiple blocks
lowfilterdelay = None
midfilterdelay = None
highfilterdelay = None


lowMomentum = 0
midMomentum = 0
highMomentum = 0


#Initializes the variables above except for sampleRate. sampleRate needs to be defined before this function is called.
def initializeAudioFilters():
    global lowcoeffs
    global midcoeffs
    global highcoeffs
    global lowfilterdelay
    global midfilterdelay
    global highfilterdelay
    global sampleRate

    #Define the frequency band separators.
    lowCutoff = 300
    midCutoff = [300, 2000]
    highCutoff = 2000
    lowNormCutoff = lowCutoff / (sampleRate / 2)
    midNormCutoff = [midCutoff[0] / (sampleRate / 2), midCutoff[1] / (sampleRate / 2)]
    highNormCutoff = highCutoff / (sampleRate / 2)

    #Initialize the global filter variables and define a low, band, and high pass filter.
    lowcoeffs = signal.butter(5, lowNormCutoff, "low")
    midcoeffs = signal.butter(5, midNormCutoff, "bandpass")
    highcoeffs = signal.butter(5, highNormCutoff, "high")

    #Initialize filter delay.
    lowfilterdelay = signal.lfilter_zi(lowcoeffs[0], lowcoeffs[1])
    midfilterdelay = signal.lfilter_zi(midcoeffs[0], midcoeffs[1])
    highfilterdelay = signal.lfilter_zi(highcoeffs[0], highcoeffs[1])




#Generates a sample sinewave for testing.
def generateSineWave(sampleRate, frequency, amplitude, duration):
    wave = np.linspace(0, 2 * np.pi * frequency * duration, sampleRate * duration)
    wave = amplitude * np.sin(wave)
    return wave


#Generates a combination of sinewaves for testing, frequencies and amplitudes should be arrays of equal length.
def generateSineWaves(sampleRate, frequencies, amplitudes, duration):
    if len(frequencies) != len(amplitudes):
        print("Error: The amount of frequencies", len(frequencies), "does not match the amount of amplitudes", len(amplitudes))
        return

    wave = generateSineWave(sampleRate, frequencies[0], amplitudes[0], duration)
    for i in range(1, len(frequencies)):
        wave = np.add(wave, generateSineWave(sampleRate, frequencies[i], amplitudes[i], duration))
    return wave

#Converts a waveform from float to integer format.
def integer_format(wave):
    intwave = np.zeros(len(wave), dtype=np.int16)
    tempwave = np.multiply(wave, 32767)
    intwave = np.add(intwave, tempwave.astype(np.int16))
    return intwave


#Converts a waveform from integer to float format.
def float_format(wave):
    floatwave = np.zeros(len(wave), dtype=np.float64)
    floatwave = np.add(floatwave, wave)
    floatwave = np.divide(floatwave, 32767)
    return floatwave

#Returns the amount of energy in the three bands for a sample size the length of the window.
#The batch of samples it analyzes are offset by the shift parameter.
def getFrequencyBandEnergies(wave, window, shift, sampleRate):

    #Perform FFT on sound. Uses shift to horizontally move the window.
    fft = np.fft.rfft(wave[shift:len(window) + shift] * window)

    #Get the frequencies and amplitudes of the sound.
    amplitudes = np.abs(fft)
    frequencies = np.fft.rfftfreq(len(window), d=1./sampleRate)

    #Filter amplitudes through a threshold
    amplitudes[amplitudes < 10.] = 0.0

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

    #Scale the magnitudes so their values become independent of the window size.
    rescale = sampleRate / len(window)
    return [lowband * rescale, midband * rescale, highband * rescale]


#Filter out band and high frequencies.
def lowPassFilter(wave, sampleRate, strength):
    global lowcoeffs
    global lowfilterdelay
    filteredWave, lowfilterdelay = signal.lfilter(lowcoeffs[0], lowcoeffs[1], wave, zi=lowfilterdelay)
    return filteredWave

#Filter out band and low frequencies.
def highPassFilter(wave, sampleRate, strength):
    global highcoeffs
    global highfilterdelay
    filteredWave, highfilterdelay = signal.lfilter(highcoeffs[0], highcoeffs[1], wave, zi=highfilterdelay)
    return filteredWave

#Filter out low and high frequencies.
def bandPassFilter(wave, sampleRate, strength):
    global midcoeffs
    global midfilterdelay
    filteredWave, midfilterdelay = signal.lfilter(midcoeffs[0], midcoeffs[1], wave, zi=midfilterdelay)
    return filteredWave

#Gets the band energies for each window block of the sound and returns it as a (n, 3) shaped python list.
def getBandEnergyArray(data, sampleRate, window):
    shift = len(window)
    numberOfSlices = len(data) // shift
    bandEnergyArray = []
    for i in range(numberOfSlices):
        bandEnergyArray.append(getFrequencyBandEnergies(data, window, shift * i, sampleRate))

    return bandEnergyArray


#Adjusts the magnitude of energy in each band to be more equalized in a window block.
def applyToneControl(data, sampleRate, bandEnergyArray, window):
    shift = len(window)
    filterData = np.zeros(len(data))

    #Go through each block and apply filters.
    for i in range(len(bandEnergyArray)):
        #Separate Bands
        lowData = lowPassFilter(data[shift * i:shift * (i + 1)], sampleRate, 5)
        midData = bandPassFilter(data[shift * i:shift * (i + 1)], sampleRate, 5)
        highData = highPassFilter(data[shift * i:shift * (i + 1)], sampleRate, 5)

        #Get the average of nonZero magnitude bands. Discarding 0s allows for better tone control and
        #avoids divide by 0 errors.
        nonZeroBandMagnitudes = []
        for j in range(3):
            if bandEnergyArray[i][j] > 0:
                nonZeroBandMagnitudes.append(bandEnergyArray[i][j])

        if len(nonZeroBandMagnitudes) == 0:
            filterData[shift * i: shift * (i + 1)] = lowData + midData + highData
            continue

        average = np.average(np.array(nonZeroBandMagnitudes))

        #Combine the bands and meaasure their energy so the program can compensate for overall sound loss.
        filteredBandEnergyArray = getFrequencyBandEnergies(lowData + midData + highData, window, 0, sampleRate)

        #Scale each band to average discarding zeros and replacing them with 1s
        lowscale = average / filteredBandEnergyArray[0] if filteredBandEnergyArray[0] != 0 else 1
        midscale = average / filteredBandEnergyArray[1] if filteredBandEnergyArray[1] != 0 else 1
        highscale = average / filteredBandEnergyArray[2] if filteredBandEnergyArray[2] != 0 else 1
        lowData = np.multiply(lowData, lowscale)
        midData = np.multiply(midData, midscale)
        highData = np.multiply(highData, highscale)


        #Combine the bands back together
        filterData[shift * i: shift * (i + 1)] = lowData + midData + highData

    #Return the tone controlled sound
    return filterData
        

def plotFrequencies(wave, window):
    x = np.linspace(0, len(window), len(window), dtype=np.int32)
    plt.plot(x, wave[:len(window)] * window)
    plt.show()


#Main entry point for the program.
if __name__ == "__main__":

    #Check to see if there are an appropriate number of arguments for the program.
    if len(sys.argv) != 2:
        print("Error: Expected 1 argument; filename")
        exit()

    #Get the filename from the command line arguments and read from the file.
    filename = sys.argv[1]
    filedata = wavfile.read(filename)

    #Global Variable. Set the sampleRate based on the sample rate in the file
    sampleRate = filedata[0]

    #Initialize the audio filters after having set the sample rate.
    initializeAudioFilters()

    #Convert sound to float format for processing.
    waveform = float_format(filedata[1])

    #Create window
    window = np.hanning(sampleRate // 10)

    #Get band energies of the sound at each window block.
    bandEnergyMagnitudes = getBandEnergyArray(waveform, sampleRate, window)

    #Apply tone control to get a new waveform.
    filteredWaveform = applyToneControl(waveform, sampleRate, bandEnergyMagnitudes, window)

    #Convert back to integer format from float.
    filteredWaveform = integer_format(filteredWaveform)

    #Write tone controlled audio to file.
    wavfile.write("toneControlled" + filename, sampleRate, filteredWaveform)
