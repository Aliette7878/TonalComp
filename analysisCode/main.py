import numpy as np
import librosa.display
import math
import matplotlib.pyplot as plt
import glob

demo_files = []
for file in glob.glob("..\\demo_sound\\*.wav"):
    demo_files.append(file)

print(demo_files)


example_number = 5
path_name = demo_files[example_number-1]
audio, Fs = librosa.load(path_name, sr=None)
print("Openning " + path_name)
print("Fs: ", Fs)

#------------------------------------------ WINDOWING ------------------------------------------
# Bandwidth given by the user
#f_low = int(input("lowest frequency = "));
#f_high = int(input("highest frequency = ")); #bandwidth = [f_low, f_high]
f_low = 120;

#Number of harmonics
N_h = 20;

#Window type
Win_type = "hamming"
#Smoothness factor
L = 4;

#Frequency separation
f_low_2=f_low*(2**(1/12)); # f_low_2 is 1/2 tone above f_low
d_fmin = 20 # to balance with the temporal resolution
d_f = max(d_fmin, int(f_low_2 - f_low));

#Window length
#Depends on our sampling frequency and the desired frequency separation
Win_length = math.floor(L*Fs/d_f);

#Number of FFT samples
#N_fft should be at least the window's length, and should respect the JND criteria
N_fft = max(2**math.ceil(math.log2(Win_length)),2**math.ceil(math.log2(Fs/(2*3))));

#Hop ratio
Hop_ratio = 4;
#Hop length
Hop_length = int(Win_length / Hop_ratio);

#Number of frames
n_frames = math.floor((len(audio) - Win_length)/Hop_length) + 1;

print("Peak resolution d_f = "+ str(d_f) + " Hz")
print("N_fft = " + str(N_fft))
print("Window length = " + str(Win_length))
print("Hop length = " + str(Hop_length))
print("n frames = " + str(math.floor((len(audio) - Win_length) / Hop_length) + 1))
print("Audio length = " + str(len(audio)))

#------------------------------------------ STFT ------------------------------------------

X = np.abs(librosa.stft(
    audio,
    win_length=Win_length,
    window=Win_type,
    n_fft=N_fft,
    hop_length=Hop_length, )
)

X_db = librosa.amplitude_to_db(X, ref=np.max)

# Frequency spectrogram


if __name__ == "__main__":  # This prevents the execution of the following code if main.py is imported in another script
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(X_db, y_axis='log', x_axis='time')

    plt.colorbar(format='%+2.0f dB')
    plt.title('Frequency spectrogram')
    plt.tight_layout()
    plt.show()

#------------------------------------------ PEAK FINDING ------------------------------------------

def peakFinding(xdB, mainPeak=False):
    n_frames = xdB.shape[1]
    peak_loc = np.zeros(n_frames)
    peak_mag = np.zeros(n_frames)

    for i in range(n_frames):
        peak_loc[i] = np.argmax(xdB[:, i])
        peak_mag[i] = xdB[int(peak_loc[i]), i]  # np.argmax return float even though here it's always int.
        if mainPeak:
            if peak_mag[i] < -25:               # just an idea to discard peak searching during silence (or low noise..)
                peak_loc[i] = peak_loc[i-1]      # silence : don't change pitch interpretation
        else:
            if peak_mag[i] < -35:               # We allow harmonics to be 10dB lower than the main peak
                peak_loc[i] = peak_loc[i-1]

    return peak_loc, peak_mag


def flattenMaxPeak(xdB, maxPeakLoc):
    n_frames = xdB.shape[1]
    for frameIndex in range(n_frames):

        minfreq = int(maxPeakLoc[frameIndex]) - int(maxPeakLoc[frameIndex] / 16)
        maxfreq = int(maxPeakLoc[frameIndex]) + int(maxPeakLoc[frameIndex] / 16)
        for freqIndex in range(minfreq, maxfreq):
            xdB[freqIndex, frameIndex] = -80
    return xdB


def movingMedian(sig, windowLength=5):
    sigSmooth = np.zeros(len(sig))
    for i in range(len(sig)):
        if i<math.floor(windowLength/2) or i>len(sig)-math.floor(windowLength/2)-1:
            sigSmooth[i] = sig[i]
        else:
            sigSmooth[i] = np.median(sig[i-math.floor(windowLength/2):i+math.floor(windowLength/2)])
    return sigSmooth


peakLoc_List = []   # Too constraining to initialise with numpy
peakMag_List = []
X_db_actualStep = X_db.copy()

numberOfPeaks = 5

for j in range(numberOfPeaks):
    if j == 0:
        Peak_loc, Peak_mag = peakFinding(X_db_actualStep, mainPeak=True)
    else:
        Peak_loc, Peak_mag = peakFinding(X_db_actualStep)
    peakLoc_List.append(Peak_loc)
    peakMag_List.append(Peak_mag)
    X_db_actualStep = flattenMaxPeak(X_db_actualStep, Peak_loc)

peakLoc_List = np.array(peakLoc_List)
peakMag_List = np.array(peakMag_List)


if __name__ == "__main__":
    plt.figure(figsize=(15, 8))
    plt.subplot(311)
    indexToFreq = Fs / N_fft

    symbolList = ['o', 'x', 'v', '*', 'h', '+', 'd', '^']
    legendList = []

    for j in range(numberOfPeaks):
        pkloc = peakLoc_List[j]
        plt.plot(np.arange(len(pkloc)), indexToFreq * pkloc, symbolList[j % len(symbolList)])
        legendList.append("peak "+str(j+1))

    plt.legend(legendList)

    plt.subplot(312)
    fundThroughFrame = np.amin(peakLoc_List, axis=0)
    plt.plot(np.arange(len(fundThroughFrame)), indexToFreq * fundThroughFrame)

    plt.subplot(313)
    N_moving_median = 5
    fundThroughFrameSmoother = movingMedian(fundThroughFrame, windowLength=N_moving_median)
    plt.plot(np.arange(len(fundThroughFrameSmoother)), indexToFreq * fundThroughFrameSmoother)

    plt.show()


