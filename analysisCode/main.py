import numpy as np
import librosa.display
import math
import matplotlib.pyplot as plt
import glob

demo_files = []
for file in glob.glob("..\\demo_sound\\*.wav"):
    demo_files.append(file)

print(demo_files)

#example_number = int(input(".wav example number = "));
example_number = 5
path_name = demo_files[example_number-1]
audio, Fs = librosa.load(path_name, sr=None)
print("Opening " + path_name)
print("Fs: ", Fs)

#------------------------------------------ WINDOWING ------------------------------------------
# Bandwidth given by the user
#f_low = int(input("lowest frequency (above 30Hz) = "));
#f_high = int(input("highest frequency = ")); #bandwidth = [f_low, f_high]
f_low = 100; #will limit d_f, shouldn't be put under 30Hz in the app

#Number of harmonics
N_h = 8;

#Window type
Win_type = "hamming"
#Smoothness factor (hop_ratio already defined above, do we need to keep L ? )
L = 4;

#Frequency separation
d_fmin = 30;   #DO NOT CHANGE - The frequency resolution is limited, to balance with the temporal resolution
d_f = max(d_fmin, int(f_low));

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
n_frames = math.floor((len(audio))/Hop_length) + 1;

print("Peak resolution d_f = "+ str(d_f) + " Hz")
print("N_fft = " + str(N_fft))
print("Window length = " + str(Win_length))
print("Hop length = " + str(Hop_length))
print("n frames = " + str(n_frames))
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

#peakFinding goes through all frames of xdB. For each frame, it keeps the loudest frequency's localization and magnitude.
#If the peak's magnitude is too low (<-25 or -35), it is interpreted as a silence. The pitch is considered as the same as the precedent frame, but silent.
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

#Erase a trajectory of xdb by setting the frequency bin around the input trajectory to -80dB.
def flattenMaxPeak(xdB, maxPeakLoc):
    n_frames = xdB.shape[1]
    for frameIndex in range(n_frames):

        minfreq = int(maxPeakLoc[frameIndex]) - int(maxPeakLoc[frameIndex] / 16)
        maxfreq = int(maxPeakLoc[frameIndex]) + int(maxPeakLoc[frameIndex] / 16)

        for freqIndex in range(minfreq, maxfreq):
            xdB[freqIndex, frameIndex] = -80
    return xdB

#Median filter on the input segment sig
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

#for each trajectory : find the maximum trajectory, save it and erase it in xdb.
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

#------------------------------------------ FUNDAMENTAL TRAJECTORY PRINTING ------------------------------------------

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

#------------------------------------------ HARMONICS FINDING ------------------------------------------
def parabolic_interpolation(alpha,beta,gamma):
    #The parabola is give by y(x) = a*(x-p)²+b where y(-1) = alpha, y(0) = beta, y(1) = gamma.
    location = 0
    value = gamma
    if alpha - 2 * beta + gamma!=0 :
        location = 0.5 * (alpha - gamma) / (alpha - 2 * beta + gamma);
        value = beta - value * (alpha - gamma) / 4;
    return [value, location]

#main lobe width
Lobe_width = math.floor(4*N_fft/Win_length)
Bw = 2*Lobe_width
Nyq = math.floor(N_fft/2);

#Where to store the harmonics magnitudes and frequency
Harmonic_db = np.zeros((n_frames,N_h))
Harmonic_freq = np.zeros((n_frames,N_h))


for n in range(n_frames):
    for h in range(2,N_h+2):

        #theorical harmonic frequency
        k_th = math.floor(h*fundThroughFrameSmoother[n])

        #draw a block around the theorical harmonic
        k_inf = max(0,k_th-Bw)
        k_inf = min(k_inf, Nyq)
        Block = X_db[k_inf:min(k_inf + 2 * Bw - 1, Nyq),n]

        maxB = max(Block)
        k_maxB =np.argmax(Block, axis=0)


        if k_maxB>0 and k_maxB<2*(Bw-1): #if k_max has adjacent samples, then interpolation is possible
            alpha = Block[k_maxB-1];
            beta  = Block[k_maxB];
            gamma = Block[k_maxB+1];
            [peak_mag, peak_loc] = parabolic_interpolation(alpha, beta, gamma); #peak_loc in [-1,1]
        else:
            [peak_mag,peak_loc]=[maxB,k_maxB]
        peak_loc = k_inf + k_maxB + peak_loc
        Harmonic_freq[n,h-2] = indexToFreq*peak_loc
        if peak_mag<-35 and n>0:
            Harmonic_db[n,h-2]=Harmonic_db[n-1,h-2]
        else:
            Harmonic_db[n,h-2] = peak_mag


if __name__ == "__main__":
    plt.figure(figsize=(15, 8))

    plt.subplot(211)
    plt.title('Harmonics frequency')
    plt.plot(np.arange(len(Harmonic_freq)), Harmonic_freq)

    plt.subplot(212)
    plt.title('Harmonics smoothed frequency')

    plt.plot(np.arange(len(fundThroughFrameSmoother)), indexToFreq * fundThroughFrameSmoother)
    Harmonic_freqSmoother = Harmonic_freq.copy()
    for h in range(2,N_h+2):
        Harmonic_freqSmoother[:,h-2] = movingMedian(Harmonic_freq[:,h-2], windowLength=N_moving_median)
    plt.plot(np.arange(len(Harmonic_freqSmoother)), Harmonic_freqSmoother)

    plt.show()

#------------------------------------------ PART 2 : SYNTEHSIS ------------------------------------------