from scipy.io import wavfile
import sounddevice as sd
import numpy as np
import torch
from pandas import read_csv
import torch.nn as nn
import torch.nn.functional as F

run_mode = "IDE"
if run_mode == "IDE":
    metadataPathName = "./code/Course Project/TinySOL_metadatamod.csv"
else:
    metadataPathName = "./TinySOL_metadatamod.csv"


def getWavfile(dataframe, index):
    return wavfile.read("./code/Course Project/TinySOL/" + dataframe['Path'].loc[dataframe.index[index]])

def getWavfileSample(dataframe, index):
    wavedata = getWavfile(dataframe, index)
    return wavedata[1][wavedata[0]:wavedata[0] * 2]

def loadSamples(dataLabels):
    data = []
    instrumentTypes = dataLabels['Instrument (abbr.)'].unique()
    labels = []
    increment = 0
    for i in range(len(dataLabels)):
        try:
            temp = getWavfileSample(dataLabels, i)
        except:
            continue
        if len(temp) == 44100:
            #print(increment, dataLabels['Instrument (abbr.)'].loc[dataLabels.index[i]], np.where(instrumentTypes == dataLabels['Instrument (abbr.)'].loc[dataLabels.index[i]]))
            labels.append(np.where(instrumentTypes == dataLabels['Instrument (abbr.)'].loc[dataLabels.index[i]]))
            data.append(temp)
            increment += 1

    return [data, labels]


def performFourierTransform(waveform):
    fft = np.fft.rfft(waveform)
    amplitudes = abs(fft)
    frequencies = np.fft.rfftfreq(len(waveform), d=1./44100)
    return [frequencies, amplitudes]

def NormalizeNdArray(array):
    normalizedArray = np.array(array, dtype=np.float32)
    normalizedArray = np.divide(normalizedArray, 32767)
    return normalizedArray

class NeuralNet1d(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 6, 6)
        self.pool = nn.MaxPool1d(2, 2)
        self.conv2 = nn.Conv1d(6, 16, 6)
        self.fc1 = nn.Linear(11021, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 14)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


if __name__ == "__main__":
    dataLabels = read_csv(metadataPathName)
    print("Retrieving Sound Data")
    data, labels = loadSamples(dataLabels)
    print("Sound Data Retrived")
    print(dataLabels['Instrument (abbr.)'].unique())
    for i in range(len(labels)):
        labels[i] = labels[i][0][0]

    for i in range(len(data)):
        data[i] = NormalizeNdArray(data[i])

    torchData = torch.from_numpy(np.array([data[0]]))
    print(torchData.dtype)
    print(torchData.shape)
    Identifier = NeuralNet1d()
    Identifier(torchData)
