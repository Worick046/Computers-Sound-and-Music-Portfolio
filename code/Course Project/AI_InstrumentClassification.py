from pdb import run
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
import sys

folderPathName = "./"


#Retrieves a wavfile from a filename at a specific index of the metadata.
def getWavfile(dataframe, index):
    return wavfile.read(folderPathName + "TinySOL/" + dataframe['Path'].loc[dataframe.index[index]])

#Retrieves a wavfile and returns a 1 second slice of audio from the file.
def getWavfileSample(dataframe, index):
    wavedata = getWavfile(dataframe, index)
    return wavedata[1][wavedata[0]:wavedata[0] * 2]

#Loads samples of each sound and their associated instrument labels
def loadSamples(dataLabels):
    data = []

    #Get an indexable array of all instrument labels in the metadata.
    instrumentTypes = dataLabels['Instrument (abbr.)'].unique()

    #Go through the metadata, load each file and add it to the return data if it is meets specifications.
    #Files need to be at least 2 seconds long so the slicing returns 1 second of audio.
    labels = []
    for i in range(len(dataLabels)):
        try:
            #Load the audio sample.
            temp = getWavfileSample(dataLabels, i)
        except:
            #If there is an error with loading an audio sample,
            #skip the sample and move to the next one.
            #At least one filename in the metadata does not exist
            #as a file so this is a necessary step.
            continue

        #If the audio sample is 1 second long, assign the appropriate label using the indexable array instrument types,
        #and add the audio sample to the returning data.
        if len(temp) == 44100:
            labels.append(np.where(instrumentTypes == dataLabels['Instrument (abbr.)'].loc[dataLabels.index[i]]))
            data.append(temp)

    return [data, labels]


#Take in audio, and perform FFT, return frequency and amplitudes.
def performFourierTransform(waveform):
    window = np.hanning(len(waveform))
    fft = np.fft.rfft(waveform * window)
    amplitudes = np.abs(fft)
    frequencies = np.fft.rfftfreq(len(waveform), d=1./44100)
    return [frequencies, amplitudes]

#Takes a integer format audio sample and converts it to float format.
def NormalizeNdArray(array):
    normalizedArray = np.array(array, dtype=np.float32)
    normalizedArray = np.divide(normalizedArray, 32767)
    return normalizedArray


#Defines the convolutional neural network
#The cnn takes in a 1 dimensional image.
class NeuralNet1d(nn.Module):
    def __init__(self):
        super().__init__()

        #Set up the convolution and fully connected layers.
        self.conv1 = nn.Conv1d(1, 6, 64)
        self.pool = nn.MaxPool1d(4, 4)
        self.conv2 = nn.Conv1d(6, 16, 64)
        self.fc1 = nn.Linear(21728, 480)
        self.fc2 = nn.Linear(480, 120)
        self.fc3 = nn.Linear(120, 14)

    #Define the feed forward process for the nn.
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

#Split data into training and testing datasets. The training data set
#is the data the neural net trains on while the testing set is used
#for assessing how well the neural net classifies new data.
#no training is done on the testing set.
def splitDataIntoTrainingAndTesting(frequencyData, labels):

    #Split data 50/50 into testing and training.

    temptrainingLabels = []
    temptestingLabels = []
    temptrainingData = []
    temptestingData = []

    #Randomly assign an audio sample to training or testing.
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

    #Randomly choose an index and add the data at that index to the assigned data category.
    #Then remove the data from the temporary list. Repeat till the temporary lists are empty.
    #Results in the randomization of the data within the training and testing sets.
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


#Load the metadata and audio samples, perform the FFT, split the data
#into training and testing, and save them into files.
def createTrainingAndTestingDatasets():
    #Retrive audio samples
    metadata = read_csv(folderPathName + "TinySOL_metadatamod.csv")
    print("Retrieving Sound data")
    wavedata, labels = loadSamples(metadata)
    print("Sound Data Retrived")

    #Convert the labels into a python list
    for i in range(len(labels)):
        labels[i] = labels[i][0][0]

    #Normalize the audio data
    for i in range(len(wavedata)):
        wavedata[i] = NormalizeNdArray(wavedata[i])


    #Perform the FFT on alll of  the audio data.
    frequencyData = []
    print("Performing FFT on dataset")
    for i in range(len(wavedata)):
        frequencyData.append(performFourierTransform(wavedata[i])[1])
    print("FFT complete")

    #Split data into training and testing, and randomize it.
    print("Randomizing Data")
    dataset = splitDataIntoTrainingAndTesting(frequencyData, labels)

    #Save the data to be used for training and testing.
    print("Saving Data")
    np.save(folderPathName + "trainingLabels.npy", dataset[0])
    np.save(folderPathName + "trainingData.npy", dataset[1])
    np.save(folderPathName + "testingLabels.npy", dataset[2])
    np.save(folderPathName + "testingData.npy", dataset[3])
    print("Data Saved")


#Load the training and testing data for model training and testing.
def loadTrainingAndTestingDatasets():
    try:
        trainingLabels = np.load(folderPathName + "trainingLabels.npy")
        testingLabels = np.load(folderPathName + "testingLabels.npy")
        trainingData = np.load(folderPathName + "trainingData.npy")
        testingData = np.load(folderPathName + "testingData.npy")
    except:
        print("Error: Could not find all required files.")
        print("Rebuilding Files")
        createTrainingAndTestingDatasets()
        trainingLabels = np.load(folderPathName + "trainingLabels.npy")
        testingLabels = np.load(folderPathName + "testingLabels.npy")
        trainingData = np.load(folderPathName + "trainingData.npy")
        testingData = np.load(folderPathName + "testingData.npy")


    #Uncompress Label data. Label data loads in as integer values.
    #Those integer values are indexes for the labels.
    #Required label format looks like [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,]
    #With the 1 being at the index specified by the integer values.
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


#Runs the model on the testing data in batches of 14. Builds the confusion matrix and
#prints out the accuracy the model got on the test. The confusion matrix is a matrix that
#contains a comparison between actual and predicted instrument classes for better analysis
#of the model's performance.
def runModelTest(testingbatches, testingLabels, numberOfTestingBatches, batchSize, identifier):
    correctPredictions = 0
    confusionMatrix = np.zeros((14, 14))
    for i in range(numberOfTestingBatches):
        #Run the model through a testing batch.
        outputs = identifier(testingbatches[i])

        #Compute accuracy of model on the current batch.
        testingBatchLabels = testingLabels[i * batchSize: i * batchSize + batchSize]
        for j in range(len(outputs)):
            if outputs[j].argmax() == testingBatchLabels[j].argmax():
                correctPredictions += 1
            confusionMatrix[outputs[j].argmax()][testingBatchLabels[j].argmax()] += 1

    print("Model accuracy: " + str(correctPredictions) + "/" + str(numberOfTestingBatches * 14))
    return confusionMatrix


#Train the model, test it, and save the model and confusion matrix.
def trainAndSaveModel():
    #Load and convert the data to float32 format.
    dataset = loadTrainingAndTestingDatasets()
    trainingLabels = torch.from_numpy(dataset[0]).to(torch.float32)
    trainingData = torch.from_numpy(dataset[1]).to(torch.float32)
    testingLabels = torch.from_numpy(dataset[2]).to(torch.float32)
    testingData = torch.from_numpy(dataset[3]).to(torch.float32)

    #Check if cuda is supported on current machine
    #cuda support allows the model to train on the gpu which accelerates
    #model training by a large factor.
    if torch.cuda.is_available():
        print("Cuda support detected, moving data to VRAM")
        device = 'cuda'
        trainingLabels = trainingLabels.to(device)
        testingLabels = testingLabels.to(device)
        trainingData = trainingData.to(device)
        testingData = testingData.to(device)
    else:
        device = 'cpu'

    #Initialize model, loss function, and optimizer.
    print("initialize model")
    identifier = NeuralNet1d().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(identifier.parameters(), lr=0.001, momentum=0.9)

    #Convert the training and testing data into batches.
    batchSize = 14
    numberOfTrainingBatches = len(trainingData) // batchSize
    numberOfTestingBatches = len(testingData) // batchSize
    trainingbatches = []
    testingbatches = []
    minibatch = []
    for i in range(len(trainingData)):
        minibatch.append(torch.stack([trainingData[i]]))
        if len(minibatch) == 14:
            trainingbatches.append(torch.stack(minibatch))
            minibatch = []

    minibatch = []
    for i in range(len(testingData)):
        minibatch.append(torch.stack([testingData[i]]))
        if len(minibatch) == 14:
            testingbatches.append(torch.stack(minibatch))
            minibatch = []

    trainingbatches = torch.stack(trainingbatches).to(device)
    testingbatches = torch.stack(testingbatches).to(device)


    #Train the model until averageLoss goes below a certain threshold.
    #model goes through the training data multiple times.
    averageLoss = 2
    while averageLoss > 0.003:
        running_loss = 0
        for j in range(numberOfTrainingBatches):
            #Reset gradients.
            optimizer.zero_grad()

            #Run forward pass on the model.
            outputs = identifier(trainingbatches[j])

            #Compute loss and perform backpropagation.
            loss = criterion(outputs, trainingLabels[j * batchSize: j * batchSize + batchSize])
            loss.backward()
            optimizer.step()

            #Compute average loss.
            running_loss += loss.item()
            if j % 100 == 99:
                averageLoss = running_loss / 100
                print(averageLoss)
                running_loss = 0


    #Run model test and build confusion matrix.
    confusionMatrix = runModelTest(testingbatches, testingLabels, numberOfTestingBatches, batchSize, identifier)
    print("Done with training")
    torch.save(identifier.state_dict(), folderPathName + "model.pth")
    np.save(folderPathName + "ConfusionMatrix.npy", confusionMatrix)

def loadModel():
    identifier = NeuralNet1d()
    identifier.load_state_dict(torch.load(folderPathName + "model.pth", weights_only=True))
    ConfusionMatrix = np.load(folderPathName + "confusionMatrix.npy")
    print(ConfusionMatrix)

if __name__ == "__main__":
    if len(sys.argv) == 2:
        if sys.argv[1] == "-t":
            trainAndSaveModel()
            exit()
        if sys.argv[1] == "-d":
            createTrainingAndTestingDatasets()
            exit()
        if sys.argv[1] == "-r":
            loadModel()
            exit()

    print("Potential Arguments are -t, -d, and -r")
    print("-t: Train and save the model")
    print("-d: Create training and testing datasets")
    print("-r: Print confusion matrix")

