from scipy.io import wavfile
import sounddevice as sd
import numpy as np
import torch
from pandas import read_csv

run_mode = "IDE"
if run_mode == "IDE":
    metadataPathName = "./code/Course Project/TinySOL_metadatamod.csv"
else:
    metadataPathName = "./TinySOL_metadatamod.csv"


def getWavfile(dataframe, index):
    return wavfile.read("./code/Course Project/TinySOL/" + dataframe['Path'].loc[dataframe.index[index]])

def getWavfileSample(dataframe, index):
    wavedata = getWavfile(dataframe, index)
    return [wavedata[0], wavedata[1][wavedata[0]:wavedata[0] * 2]]

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
        if len(temp[1]) == 44100:
            print(increment, dataLabels['Instrument (abbr.)'].loc[dataLabels.index[i]], np.where(instrumentTypes == dataLabels['Instrument (abbr.)'].loc[dataLabels.index[i]]))
            labels.append(np.where(instrumentTypes == dataLabels['Instrument (abbr.)'].loc[dataLabels.index[i]]))
            data.append(temp)
            increment += 1

    return [data, labels]


if __name__ == "__main__":
    dataLabels = read_csv(metadataPathName)
    print("Retrieving Sound Data")
    data, labels = loadSamples(dataLabels)
    print("Sound Data Retrived")
    print(dataLabels['Instrument (abbr.)'].unique())
    for i in range(len(labels)):
        labels[i] = labels[i][0][0]

    print(labels.index(1))
    sd.play(data[labels.index(1) - 1][1], data[labels.index(1) - 1][0])
    sd.wait()