import sounddevice as sd
import numpy as np
from scipy.io import wavfile
import time


#Generates a sine wave using frequency, amplitude, phase, and duration.
def generateWave(frequency, amplitude, phase, duration):
    sampleRate = 44100
    wave = np.linspace(0, 1 * duration, 44100 * duration, dtype = np.float64)
    wave = np.sin(2 * np.pi * frequency * wave + phase)
    wave = np.multiply(wave, amplitude)
    return wave

#Computes the Discrete Fourier Transform
def computeDFT(data):
    X = np.fft.rfft(data)
    return X

#Computes the amplitudes of the DFT
def computeAmplitudes(data):
    X = np.abs(data)
    return X

#Computes the phases of the DFT
def computePhases(data):
    X = np.angle(data)
    return X

#A representation of the data needed to create a sine wave
class tone:
    def __init__(self, frequency, amplitude, phase):
        self.frequency = frequency
        self.amplitude = amplitude
        self.phase = phase

#Removes frequencies that are too quiet or not a local maximum.
def filterFrequencies(tones):
    maxFound = False
    filteredtones = []
    #Go through all the tones.
    for i in range(len(tones)):
        lastFrequency = i - 1
        #Skip over frequencies that are too quiet.
        if lastFrequency - 1 == -1 or tones[i].amplitude < 1:
            continue
        
        #Filters out frequencies that are not at a local maximum.
        if tones[i].amplitude > tones[lastFrequency].amplitude and maxFound == True:
            maxFound = False
        elif tones[i].amplitude < tones[lastFrequency].amplitude and maxFound == False:
            maxFound = True
            filteredtones.append(tones[lastFrequency])
    return filteredtones


#Compute tone data from a sound sample.
def computeTones(data):
    #Adds zero padding to increase the resolution of the DFT
    desiredLength = pow(2, 20)
    data = np.pad(data, (0, desiredLength - len(data)), 'constant')

    #Compute the DFT and then compute the amplitudes and phases.
    F = computeDFT(data)
    amplitudes = computeAmplitudes(F)
    phases = computePhases(F)
    tones = []

    #Convert the data gathered into a list of the tone class.
    for i in range(len(amplitudes)):
        tones.append(tone(i, amplitudes[i], phases[i]))

    #Filter the tones.
    tones = filterFrequencies(tones)
    return tones


#Rescales frequency after using zero padding to increase the resolution
def scaleFrequency(tones):
    for i in range(len(tones)):
        tones[i].frequency = 44100/pow(2, 20) * tones[i].frequency

    return tones

#Scales amplitude to more accurate levels.
def scaleAmplitudes(tones):
    for i in range(len(tones)):
        tones[i].amplitude /= (44100 / 2)

    return tones

#Takes a list of tones and a duration and outputs a waveform from the tone data
def generateWaveFromTones(tones, duration):
    wave = generateWave(tones[0].frequency, tones[0].amplitude, tones[0].phase, duration)
    for i in range(1, len(tones)):
        wave = np.add(wave, generateWave(tones[i].frequency, tones[i].amplitude, tones[i].phase, duration))

    return wave

#Finds the fundamental note of the sound.
def findFundamental(tones):
    maxTone = tones[0]
    for i in range(1, len(tones)):
        if tones[i].amplitude > maxTone.amplitude:
            maxTone = tones[i]
    return maxTone

#Clusters tones of very similar frequencies together.
def findCluster(frequency, tones):
    clusterTones = []
    for i in range(len(tones)):
        if abs(frequency - tones[i].frequency) <= 0.1:
            clusterTones.append(tones[i])
    return clusterTones

#Clusters tones together and chooses the one with the largest amplitude, discarding the rest of the tones in the cluster.
#Returns a list of 1 tone per cluster.
def decluster(tones):
    fundamental = findFundamental(tones)
    clusters = []
    for i in range(30):
        clusters.append(findCluster(fundamental.frequency * (i + 1), tones))

    declusteredTones = []
    for i in range(len(clusters)):
        if len(clusters[i]) == 0:
            continue
        
        maxTone = clusters[i][0]
        for j in range(1, len(clusters[i])):
            if maxTone.amplitude < clusters[i][j].amplitude:
                maxTone = clusters[i][j]
        declusteredTones.append(maxTone)
    return declusteredTones

#Removes all frequencies that are below the fundamental note.
def removeEverythingBelowFundamental(tones):
    fundamental = findFundamental(tones)
    while tones[0].frequency < fundamental.frequency:
        tones.pop(0)
    return tones

#Takes a waveform and computes tone data.
def generateTonesFromWave(waveform):
    tones = computeTones(waveform)
    tones = scaleFrequency(tones)
    tones = scaleAmplitudes(tones)

    #The commented out functions below filter out a lot of frequencies which results in an interesting sound,
    #but you will get more accuracy with replication without them. 
    #tones = removeEverythingBelowFundamental(tones)
    #tones = decluster(tones)
    return tones

#Reads a wavfile and extracts a 1 second sample from it.
def readWav(filename):
    wave = wavfile.read(filename)
    waveSample = wave[1].astype(np.float64)
    waveSample = np.divide(wave[1], pow(2, 16))
    waveSample = waveSample[44100:44100 * 2]
    return waveSample


#Program entry point
if __name__ == "__main__":
    #Get 1 second sample of cello.
    celloWave = readWav("./code/Sound Reconstructor/Vc-ord-G3-mf-2c-N.wav")
    tones = generateTonesFromWave(celloWave)

    #Print the fundamental frequency.
    print("Fundamental:", findFundamental(tones).frequency)

    print("Original sound")
    sd.play(celloWave, 44100)
    sd.wait()
    newWave = np.multiply(generateWaveFromTones(tones, 1), 2)
    print("Reconstructed sound")
    sd.play(newWave, 44100)
    sd.wait()
