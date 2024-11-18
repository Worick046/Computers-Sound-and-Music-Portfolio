import numpy as np
import sounddevice as sd
from scipy.io import wavfile
import math


#Converts single channel data into 2 channel data
def convertMonoToStereo(data):
    #Allocate space for 2 channels
    newData = np.zeros((len(data[1]), 2), dtype=np.int16)

    #Copy data from single channel to 2 channels
    for i in range(len(data[1])):
        newData[i][0] = data[1][i]
        newData[i][1] = data[1][i]

    #Return sample rate and data as tuple
    return (data[0], newData)


#Takes audio data in the form of a tuple (sample rate, data) and inverted frequency which is 1 / frequency or the amount of seconds
#it takes to complete 1 revolution. Scales amplitude on 2 channels using a sine wave to oscillate the volume between the channels.
def oscillate(Data, invertedFrequency):
    #If the data is single channel, convert to dual channel.
    if len(Data[1].shape) == 1:
        print("Converting Mono to Stereo")
        Data = convertMonoToStereo(Data)
    sampleRate = Data[0]
    #Oscillate amplitude between the channels
    print("Calculating stereo oscillation with inverted frequency of", invertedFrequency, "seconds")
    for i in range(len(Data[1])):
        scale = 0.35 * math.sin(math.pi * float(i) / (invertedFrequency / 2 * sampleRate)) + 0.5
        Data[1][i][0] *= 1 - scale
        Data[1][i][1] *= scale
    return Data

#Program entry point
if __name__ == "__main__":
    #Define file paths
    filepath1 = "./code/Stereo Oscillation/Oscillations.wav"
    filepath2 = "./code/Stereo Oscillation/Oscillationsmono.wav"
    destinationPath1 = "./code/Stereo Oscillation/OscillationsO.wav"
    destinationPath2 = "./code/Stereo Oscillation/OscillationsmonoO.wav"

    #Read from wav files
    soundfilestereo = wavfile.read(filepath1)
    soundfilemono = wavfile.read(filepath2)

    #Convert audio data
    print("Converting", filepath1)
    stereoSound = oscillate(soundfilestereo, 16)
    print("Converting", filepath2)
    monoSound = oscillate(soundfilemono, 16)

    #Write to new wav files
    wavfile.write(destinationPath1, stereoSound[0], stereoSound[1])
    wavfile.write(destinationPath2, monoSound[0], monoSound[1])
    print("Conversion complete")

    #Play first converted audio
    print("Playing", destinationPath1)
    sd.play(stereoSound[1], stereoSound[0])
    sd.wait()
    
