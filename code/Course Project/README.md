# Course Final Project
## Instrument classification using artificial intelligence and Fourier Transforms

## Description
This project is an experiment in using machine learning to analyze audio and
classify what instrument produced that audio. The model takes in an amplitude image
of the fourier transform and outputs 1 of 14 insruments that could have produced
the sound.

### Audio Data
The audio data I am using for this experiment is from the TinySOL instrument library
(license below) which contains audio data from 14 different instruments with each
instrument having multiple wav files of that instrument playing different notes at
different volumes and different durations.

### ML Model
The model I used is called a convolutional neural network which works by using
filters to extract features from an image. The process involves passing each filter
over the entire image which creates a set of new images. Then the process repeats for
each convolution layer in the network (I made 2). After the data goes through the
convolution layers it then gets flattened into a 1 dimensional array which then gets
passed into a multilayer perceptron. The multilayer perceptron is what is usually shown
as a visual for neural networks where there is multiple layers of artificial neurons that
connect with each other.

### How the model was used
In order to use the model I made I had to turn the audio data from the TinySOL instrument library
into images that could be processed. Luckily, the convolutional neural network works on more than
just 2 dimensional images. While the audio data itself technically counts as a 1 dimensional image,
I thought it would be to messy to work with directly so I used an FFT to get a 1 dimensional image
of frequency magnitudes that I could then train on. My biggest reason for choosing the
convolutional neural network was the flexibility of identifying features no matter where they are
in the image. Since each filter gets applied to every part of the image, the appropriate features
are discovered. This is important because an instrument playing different notes will shift the
features of that image left or right depending on if the note played was lower or higher.

### Results
The confusion matrix shows a comparison between the actual class and predicted class of instrument.
In order the instruments are
Bass Tuba,
French Horn,
Trombone,
Trumpet in C,
Accordion,
Cello,
Contrabass,
Viola,
Violin,
Alto Saxophone,
Bassoon,
Clarinet in B-flat,
Flute, and
Oboe

The actual class of the instrument is the row while the predicted class is the column.


|     | BTb | Hn | Tbn | TPc | Acc | Vc | Cb | Va | Vn | ASax | Bn | ClBb | Fl | Ob |
|-----|-----|----|-----|-----|-----|----|----|----|----|------|----|------|----|----|
| Btb |  29 |  0 |   0 |   0 |  0  |  0 | 0  | 0  | 0  |  1   | 0  |  0   | 0  |  0 |
| Hn  |  13 | 52 |   2 |   0 |  1  |  0 | 0  | 0  | 0  |  4   | 5  |  1   | 0  |  0 |
| Tbn |  0  |  2 |  50 |   0 |  1  |  0 | 0  | 0  | 0  |  0   | 6  |  2   | 0  |  0 |
| TPc |  0  |  0 |   0 |  32 |  0  |  0 | 0  | 1  | 0  |  0   | 1  |  1   | 1  | 13 |
| Acc |  0  |  0 |   3 |   0 | 335 |  0 | 1  | 0  | 3  | 11   | 0  |  4   | 3  |  1 |
| Vc  |  0  |  0 |   0 |   0 |  0  |142 | 2  | 3  | 0  |  0   | 0  |  0   | 0  |  0 |
| Cb  |  0  |  0 |   0 |   1 |  0  |  3 |135 | 0  | 0  |  0   | 1  |  0   | 0  |  0 |
| Va  |  0  |  0 |   0 |   0 |  0  |  4 | 0  |150 | 1  |  1   | 0  |  1   | 2  |  1 |
| Vn  |  0  |  0 |   0 |   2 |  0  |  1 | 0  | 3  |139 |  0   | 0  |  0   | 0  |  3 |
| ASax|  1  |  1 |   4 |   0 |  1  |  0 | 0  | 1  | 0  | 21   | 0  |  5   | 1  |  0 |
| Bn  |  0  |  5 |   8 |   0 |  0  |  0 | 0  | 0  | 0  |  0   |56  |  1   | 0  |  0 |
| ClBb|  0  |  0 |   0 |   1 |  0  |  0 | 0  | 0  | 0  |  7   | 1  | 24   |11  |  2 |
| Fl  |  0  |  1 |   0 |   0 |  2  |  1 | 0  | 0  | 0  |  3   | 1  | 10   |33  |  5 |
| Ob  |  0  |  0 |   0 |   6 |  0  |  0 | 0  | 0  | 0  |  0   | 0  |  5   | 3  | 33 |

For the most part the model was successful in classifying each sound correctly with an
86.2% accuracy. The most interesting misclassification to me was predicting 13 French Horn
sounds to be produced by the Bass Tuba. What is interesting about this is that none of the
Bass Tuba sounds were misidentified as a French Horn which means that this was a 1 way
misclassification. Some other notable misclassifications are flute sounds being labeled clarinet,
Trumpet sounds being labeled as Oboe, and clarinet sounds being labeled as flute. It seems the
model had a little trouble with the flute and clarinet as quite a significant portion of their
sounds seem to be predicted as the other's sound. The last noteable misclassification is between
Alto Sax and Accordion. where 11 Accordion sounds where identified as Alto Sax and 1 Alto Sax sound
was identified as an Accordion.

### Future impovement
In this experiment the model was decently successful with classifying instruments correctly most of the time.
An avenue for future improvement could be to include a temporal component in the data and if a model using
time as well as frequency would do better. That is, this model uses a snapshot of frequency but it would be
interesting to see how a model performs if the input data is something like a spectrogram. Once an accurate
enough model is made it would be interesting to combine that with note identification to build XML files
from audio. (Note: This would not be made for the purpose of pirating sheet music and I do not condone such
action)


## References used

How to build a convolutional neural network(cnn) for 2d images. I followed this for
a refresh on how to build one and built
a cnn for a 1 dimensional image.
https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

## License
TinySOL is provided under the following license
https://creativecommons.org/licenses/by/4.0/

Details of TinySOL can be found here including authors

https://zenodo.org/records/3685331


