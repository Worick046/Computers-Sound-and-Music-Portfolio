#Author: Patrick Wood
#Description: This program generates 2 .wav files called sine.wav and clipped.wav and plays a clipped wave at
#440HZ
#sine.wav is a sinewave at 440HZ
#clipped.wav is a clipped sine wave at 440HZ

import sounddevice as sd
from scipy.io import wavfile
import math
import numpy as np


def generateSinewave(frequency):
    t = np.linspace(0, 1, 48000, dtype=np.float64)
    wave = np.sin(2 * np.pi * frequency * t)
    wave = wave * 32767 * 0.25
    wave = wave.astype(np.int16)
    wavfile.write("sine.wav", 48000, wave)
    return wave

def generateClippedWave(frequency):
    time = np.linspace(0, 1, 48000, dtype=np.float64)
    wave = np.sin(2 * np.pi * frequency * time)
    wave = wave * 32767 * 0.5
    threshold = 32767 / 4
    wave = np.clip(wave, -threshold, threshold)
    wave = wave.astype(np.int16)
    wavfile.write("clipped.wav", 48000, wave)
    return wave

if __name__ == '__main__':
    generateSinewave(440)
    sd.play(generateClippedWave(440), 48000)
    sd.wait()