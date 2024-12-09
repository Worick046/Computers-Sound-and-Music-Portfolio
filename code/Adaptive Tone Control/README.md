# Adaptive Tone Control

## Description
Equalize the energy in low frequency(1 - 300), mid frequency(300 - 2000), and high frequency(2000 and up) ranges
using an fft for measurement and audio filters to adjust the intensity of each range. Analyzes and adjusts small blocks
of samples at a time to dynamically change the frequency magnitude adjustment throughout the piece. The adjustments
are made by running the blocks through a low, band, and high pass filter scaling the audio outputted from the filter,
and adding them back up into a new wave.

Arguments:
Takes a filename as a command line argument. Must be a wav file. Please leave out the ./ in the filename

Output:
Generates a tone controlled file with a filename of "toneControlled" + the original file name


## What I Learned
This was a very challenging project for me and it is still not as effective as I would like it to be. I
learned a lot about how to do smooth filtering over multiple blocks without introducing artifacts, the nature
of windowing functions, a better understanding of how to use ffts, and the utilization of more features in the numpy module.

## For the Future
One thing I think would be cool to implement and would improve the filtering would be a momentum variable.
Right now the program can introduce volume artifacts, but a momentum variable would limit how fast it can
adjust tones based on previous adjustments which would smooth out the volume artifacts.

