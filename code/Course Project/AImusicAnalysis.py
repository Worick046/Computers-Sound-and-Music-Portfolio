from scipy.io import wavfile
import sounddevice as sd
import numpy as np
import torch
from pandas import read_csv
import torch.nn as nn
import torch.nn.functional as F
import random

run_mode = "IDE"
if run_mode == "IDE":
    folderPathName = "./code/Course Project/"
else:
    folderPathName = "./"


def getWavfile(dataframe, index):
    return wavfile.read(folderPathName + "TinySOL/" + dataframe['Path'].loc[dataframe.index[index]])

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
    window = np.hanning(len(waveform))
    fft = np.fft.rfft(waveform * window)
    amplitudes = np.abs(fft)
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
        self.fc1 = nn.Linear(5509, 120)
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


def splitDataIntoTrainingAndTesting(frequencyData, labels):

    #Split data 50/50 into testing and training.

    temptrainingLabels = []
    temptestingLabels = []
    temptrainingData = []
    temptestingData = []

    for i in range(len(frequencyData)):
        if random.randint(0, 1) == 1:
            temptrainingLabels.append(labels[i])
            temptrainingData.append(frequencyData[i])
        else:
            temptestingLabels.append(labels[i])
            temptestingData.append(frequencyData[i])


    #Randomize the data within their respective categories.

    trainingLabels = []
    testingLabels = []
    trainingData = []
    testingData = []

    while(len(temptrainingData) > 0):
        randomIndex = random.randint(0, len(temptrainingData) - 1)
        trainingData.append(temptrainingData[randomIndex])
        trainingLabels.append(temptrainingLabels[randomIndex])
        temptrainingData.pop(randomIndex)
        temptrainingLabels.pop(randomIndex)

    while(len(temptestingData) > 0):
        randomIndex = random.randint(0, len(temptestingData) - 1)
        testingData.append(temptestingData[randomIndex])
        testingLabels.append(temptestingLabels[randomIndex])
        temptestingData.pop(randomIndex)
        temptestingLabels.pop(randomIndex)

    return [np.array(trainingLabels), np.array(trainingData), np.array(testingLabels), np.array(testingData)]


def createTrainingAndTestingDatasets():
    metadata = read_csv(folderPathName + "TinySOL_metadatamod.csv")
    print("Retrieving Sound data")
    wavedata, labels = loadSamples(metadata)
    print("Sound Data Retrived")
    for i in range(len(labels)):
        labels[i] = labels[i][0][0]

    for i in range(len(wavedata)):
        wavedata[i] = NormalizeNdArray(wavedata[i])

    frequencyData = []
    print("Performing FFT on dataset")
    for i in range(len(wavedata)):
        frequencyData.append(performFourierTransform(wavedata[i])[1])
    print("FFT complete")
    print("Randomizing Data")
    dataset = splitDataIntoTrainingAndTesting(frequencyData, labels)
    print("Saving Data")
    np.save(folderPathName + "trainingLabels.npy", dataset[0])
    np.save(folderPathName + "trainingData.npy", dataset[1])
    np.save(folderPathName + "testingLabels.npy", dataset[2])
    np.save(folderPathName + "testingData.npy", dataset[3])
    print("Data Saved")


if __name__ == "__main__":


    #Identifier = NeuralNet1d()
    #Identifier(dataset[0:1])
