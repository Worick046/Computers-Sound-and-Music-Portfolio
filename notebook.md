# Notebook

This notebook contains a chronological account of my work and progress in Computers, Sounds, and Music.


## 10/8/2024
I have started and completed project 1, the Clipped Sine Wave which involved learning how to generate playable
sine waves and clamping the maximum and minimum values. I have installed sounddevice on my computer to play
the audio data that I generate. The next thing I need to work on is filling in the main README.md file and
the README.md for the project.


## 10/12/2024
Filled out the README.md to introduce the repository and Filled out the Clipped Sine Wave README.md
To describe the project and what I learned.

## 10/19/2024
Introduced myself on Zulip, the messaging platform the class is using. I said my name, my background in music
and expressed my excitement for the class. My exact words are below.

Hi everyone! My name is Patrick. I am an undergrad CS major and a Music minor. I am a member of the Portland
State Chamber Choir and take voice lessons through the music department. I've always been interested in the
combination of computers and music, and have recorded myself singing multiple parts of choral pieces to put
them together in a DAW. I am very excited to learn how audio works at a lower level and hopefully make some
cool sounds!

## 10/22/2024
Made a new project called DFT to understand how to perform the Discrete Fourier Transform using Python libraries.

## 10/26/2024
Using the skills I learned from the DFT project I made a new project called Instrument Emulator in which I
further explored how to utilize the FFT by increaseing frequency resolution, and utilizing resuting
amplitudes. I will be testing on instrument sounds that are under a creative commons license next.

## 11/16/2024
I have renamed the Instrument Emulator project to Sound Reconstructor as I realized the 
scope of the project I was trying to do was too large to feasibly complete within a reasonable amount
of time. I finished the Sound Reconstructor project in which I learned more about the limits of DFT in which
the resulting data is much more dependent on the original sound than I had realized. I successfully recreated
the sound of a cello with pretty good accuracy.

## 11/17/2024
I learned how to work with stereo audio in python. I thought it was going to be 2 arrays of samples,
but it was actually 1 large array with many size 2 arrays and each of the small arrays contained 1 sample
per channel. I used a piece I composed recently to test the program out and applied the 8D effect to it.

## 11/25/2024
Started Adaptive Tone Control Project. Learned how to use a low pass filter and set up a frequency band measuring
function.

## 12/8/2024
Finished up the Adaptive Tone Control Project. I have a much better understanding of windowing functions, audio
filters, measuring frequency, and the amount of little mistakes that can completely change the resulting waveform.
This was a challenging project for me. I understand better that windowing functions can act as a low pass filter themselves,
and that in order to have a smooth filtering experience when you are applying a filter multiple times across a waveform,
you need to keep track of the filter delay values so the filter can pick up where it left off instead of using it's starting
vectors every time as that introduces an interesting and undesired effect.

## 12/9/2024
Started and finished the popgen project. I took the popgen program written by Bart Massey and modified it to add harmonics
to the notes as they were just pure sine waves, I locked the melody so it does not go too high and become irritating and also
does not go too low and mix with the bass chords. Last thing I did was add a rhythm generator that generates random rythms
consisting of quarter and eighth notes.


## 12/10/2024
I built an AI model to classify instruments based on audio data. The model was a success and I was able to get a 86.2% accuracy
in classifying the data. I wanted to try my hand at audio analysis for the final project because I have a great interest
in data analysis. Through building this program I learned so much about how important context is to audio analysis.
In some ways I was only able to get to a 86.2% accuracy. I used a model that only looked at frequencies in a snapshot
of time. I have read that other models can be built to include how the frequencies change over time. It may or may
not work too well on instruments that are playing a continuous sound as I don't imagine that would have a large amount
of frequency change over time but maybe the start and end of an instruments sound are important. 

## 12/11/2024
I have finished up the final course project for AI instrument classification including a more in depth comparison on
how the model did with specific instruments. I made it able to be run from the command line as well as giving that
capability to my other programs for grading purposes.
