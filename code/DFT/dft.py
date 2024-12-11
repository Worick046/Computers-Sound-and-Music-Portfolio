import sounddevice as sd
import scipy.io.wavfile as wavfile
import math
import numpy as np
from scipy.fft import fft

def generateWave(frequency, amplitude, phase):
    sampleRate = 48000
    duration = 1
    wave = np.linspace(0, 1, 48000, dtype = np.float64)
    wave = np.sin(2 * np.pi * frequency * wave + phase)
    wave = np.multiply(wave, amplitude)
    return wave


def computeK(frequency, samples, sampleRate):
    k = frequency * samples / sampleRate
    return k


def computeDFT(data):
    X = np.fft.rfft(data)
    return X

def computeMagnitudes(data):
    X = np.abs(data)
    return X

def computePhases(data):
    X = np.angle(data)
    return X

if __name__ == "__main__":
    #Generate Initial Wave
    wave = generateWave(440, 0.2, 0)
    wave = np.add(wave, generateWave(880, 0.2, 0))
    wave = np.add(wave, generateWave(1760, 0.2, 0))
    #Play initial wave
    sd.play(wave)
    sd.wait()
    wavfile.write("./wave.wav", 48000, wave)

    #Perform DFT and reconstruct wave
    fwave = computeDFT(wave)

    #frequency magnitudes
    mWave = np.abs(fwave)

    #Frequency phases
    waveAngle = np.angle(fwave)

    #Reconstruct wave
    newWave = np.zeros(48000)
    for i in range(len(mWave) - 1):
        if(mWave[i] > 200):
            newWave = np.add(newWave, generateWave(i, 0.2, waveAngle[i]))

    sd.play(newWave)
    sd.wait()
    wavfile.write("./reconstructedWave.wav", 48000, wave)
    
