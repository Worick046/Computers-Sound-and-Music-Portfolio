from scipy.io import wavfile
import sounddevice as sd
import numpy as np
import torch
from pandas import read_csv
from torch.cuda import is_available
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
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
        self.conv1 = nn.Conv1d(1, 6, 1000)
        self.pool = nn.MaxPool1d(2, 2)
        self.conv2 = nn.Conv1d(6, 10, 1000)
        self.fc1 = nn.Linear(47630, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 14)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x)
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


def loadTrainingAndTestingDatasets():
    try:
        trainingLabels = np.load(folderPathName + "trainingLabels.npy")
        testingLabels = np.load(folderPathName + "testingLabels.npy")
        trainingData = np.load(folderPathName + "trainingData.npy")
        testingData = np.load(folderPathName + "testingData.npy")
    except:
        print("Error: Could not find all required files")
        print("Files Needed")
        print("trainingLabels.npy")
        print("testingLabels.npy")
        print("trainingData.npy")
        print("testingData.npy")

    #Uncompress Label data.
    uncompressedTrainingLabels = []
    uncompressedTestingLabels = []
    for i in range(len(trainingLabels)):
        newLabel = np.zeros(14)
        newLabel[trainingLabels[i]] = 1
        uncompressedTrainingLabels.append(newLabel)
    for i in range(len(testingLabels)):
        newLabel = np.zeros(14)
        newLabel[testingLabels[i]] = 1
        uncompressedTestingLabels.append(newLabel)

    trainingLabels = np.array(uncompressedTrainingLabels)
    testingLabels = np.array(uncompressedTestingLabels)
    
    return [trainingLabels, trainingData, testingLabels, testingData]


def trainAndSaveModel():
    dataset = loadTrainingAndTestingDatasets()
    trainingLabels = torch.from_numpy(dataset[0]).to(torch.float32)
    trainingData = torch.from_numpy(dataset[1]).to(torch.float32)
    testingLabels = torch.from_numpy(dataset[2]).to(torch.float32)
    testingData = torch.from_numpy(dataset[3]).to(torch.float32)

    if torch.cuda.is_available():
        print("Cuda support detected, moving data to VRAM")
        device = 'cuda'
        trainingLabels = trainingLabels.to(device)
        testingLabels = testingLabels.to(device)
        trainingData = trainingData.to(device)
        testingData = testingData.to(device)
    else:
        device = 'cpu'

    print("initialize model")
    identifier = NeuralNet1d().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(identifier.parameters(), lr=0.001, momentum=0.9)

    number_of_epochs = 10
    for i in range(number_of_epochs):
        running_loss = 0
        for i in range(len(trainingData)):
            optimizer.zero_grad()
            outputs = identifier(trainingData[i:i + 1])
            loss = criterion(outputs, trainingLabels[i])
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 99:
                print(running_loss / 100)
                running_loss = 0


    correctPredictions = 0
    for i in range(len(testingData)):
        outputs = identifier(testingData[i:i + 1])
        if outputs.argmax() == testingLabels.argmax():
            correctPredictions += 1

    print(str(correctPredictions) + "/" + str(len(testingData)))
    torch.save(identifier.state_dict(), folderPathName + "model.pth")


if __name__ == "__main__":
    identifier = NeuralNet1d()
    identifier.load_state_dict(torch.load(folderPathName + "model.pth", weights_only=True))
    
    dataset = loadTrainingAndTestingDatasets()
    trainingLabels = torch.from_numpy(dataset[0]).to(torch.float32)
    trainingData = torch.from_numpy(dataset[1]).to(torch.float32)
    testingLabels = torch.from_numpy(dataset[2]).to(torch.float32)
    testingData = torch.from_numpy(dataset[3]).to(torch.float32)

    if torch.cuda.is_available():
        print("Cuda support detected, moving data to VRAM")
        device = 'cuda'
        trainingLabels = trainingLabels.to(device)
        testingLabels = testingLabels.to(device)
        trainingData = trainingData.to(device)
        testingData = testingData.to(device)
    else:
        device = 'cpu'

    identifier = identifier.to(device)



            



