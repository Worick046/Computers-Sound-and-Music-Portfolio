from scipy.io import wavfile
import sounddevice as sd
import numpy as np
import torch
from pandas import read_csv
import mirdata

run_mode = "IDE"
if run_mode == "IDE":
    metadataPathName = "./code/Course Project/TinySOL_metadatamod.csv"
else:
    metadataPathName = "./TinySOL_metadatamod.csv"


def getWavfile(dataframe, index):
    return wavfile.read("./code/Course Project/TinySOL/" + dataframe['Path'].loc[dataframe.index[index]])

if __name__ == "__main__":
    dataLabels = read_csv(metadataPathName)
    wavedata = getWavfile(dataLabels, 400)
    print(wavedata)
    sd.play(wavedata[1], wavedata[0])
    sd.wait()