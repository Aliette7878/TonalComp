import librosa.display
import math
import numpy as np
import matplotlib.pyplot as plt


class AudioAnalysis:

    def __init__(self, pathName, analysisParams):
        # self.analysisParams = analysisParams      #for the moment not needed outside
        print("Opening " + pathName)
        self.pathName = pathName
        self.audio, self.Fs = librosa.load(pathName, sr=None)       # mp3 not supported yet (can be with ffmpeg)
        print("Fs: ", self.Fs)
        self.win_length = math.floor(analysisParams.L * self.Fs / analysisParams.d_f)
        self.N_fft = max(2 ** math.ceil(math.log2(self.win_length)), 2 ** math.ceil(math.log2(self.Fs / 3)))
        self.indexToFreq = self.Fs / self.N_fft
        self.hop_length = int(self.win_length / analysisParams.hop_ratio)

        self.n_frames = math.floor((len(self.audio)) / self.hop_length) + 1

        self.X = np.abs(librosa.stft(
            self.audio,
            win_length=self.win_length,
            window=analysisParams.win_type,
            n_fft=self.N_fft,
            hop_length=self.hop_length, )
        )
        self.X_db = librosa.amplitude_to_db(self.X, ref=np.max)

        print("Peak resolution d_f = " + str(analysisParams.d_f) + " Hz")
        print("N_fft = " + str(self.N_fft))
        print("Window length = " + str(self.win_length))
        print("Hop length = " + str(self.hop_length))
        print("n frames = " + str(self.n_frames))
        print("Audio length = " + str(len(self.audio)))

    def showframe(self, frameIndex):
        fig2 = plt.figure(figsize=(20, 15))
        plt.plot(self.indexToFreq * np.arange(self.X_db[:1500, frameIndex].size), self.X_db[:1500, frameIndex])
        fig2.show()

    def showpeaks(self, numberOfPeaks):
        peakLoc_List = []  # Too constraining to initialise with numpy
        # peakMag_List = []
        X_db_actualStep = self.X_db.copy()

        # for each trajectory : find the maximum trajectory, save it and erase it in xdb.
        for j in range(numberOfPeaks):
            if j == 0:
                Peak_loc, Peak_mag = peakFinding(X_db_actualStep, mainPeak=True)
            else:
                Peak_loc, Peak_mag = peakFinding(X_db_actualStep)
            peakLoc_List.append(Peak_loc)
            # peakMag_List.append(Peak_mag)
            X_db_actualStep = flattenMaxPeak(X_db_actualStep, Peak_loc)

        peakLoc_List = np.array(peakLoc_List)
        # peakMag_List = np.array(peakMag_List)

        N_moving_median = 5
        fundThroughFrame = np.amin(peakLoc_List, axis=0)    # Depending on the number of peaks computed, this can be
        # irrelevant (too much peak = peaks under fundamental, only 1 or 2 peaks can catch harmonics only)
        fundThroughFrameSmoother = movingMedian(fundThroughFrame, windowLength=N_moving_median)

        plt.figure(figsize=(15, 8))

        plt.subplot(311)
        symbolList = ['o', 'x', 'v', '*', 'h', '+', 'd', '^']
        legendList = []
        for j in range(numberOfPeaks):
            pkloc = peakLoc_List[j]
            plt.plot(np.arange(len(pkloc)), self.indexToFreq * pkloc, symbolList[j % len(symbolList)])
            legendList.append("peak " + str(j + 1))
        plt.legend(legendList)

        plt.subplot(312)
        plt.plot(np.arange(len(fundThroughFrame)), self.indexToFreq * fundThroughFrame)

        plt.subplot(313)
        plt.plot(np.arange(len(fundThroughFrameSmoother)), self.indexToFreq * fundThroughFrameSmoother)
        plt.show(block=False)


class AnalysisParameters:

    d_fmin = 30
    hop_ratio = 4

    def __init__(self):
        self.f_low = 50
        self.N_h = 10
        self.win_type = "hamming"   # Window type
        self.L = 4                  # Smoothness factor of the chosen window's type
        self.d_f = self.f_low
        if self.f_low < self.d_fmin:
            self.d_f = self.d_fmin
            print('\033[93m' + f"WARNING: f_low chosen too low, and therefore changed to {self.d_fmin}")


# -------------------------------------------- PEAK TRACKING FUNCTIONS--------------------------------------------------

# peakFinding goes through all frames of xdB. For each frame, it keeps the peaks localization (freq) and magnitude.
# If the peak's magnitude is too low (<-25 or -35), it is interpreted as a silence.
# The pitch is considered as the same as the precedent frame, but silent.
def peakFinding(xdB, mainPeak=False):
    n_frames = xdB.shape[1]
    peak_loc = np.zeros(n_frames)
    peak_mag = np.zeros(n_frames)

    for i in range(n_frames):
        peak_loc[i] = np.argmax(xdB[:, i])
        peak_mag[i] = xdB[int(peak_loc[i]), i]  # np.argmax return float even though here it's always int.
        if mainPeak:
            if peak_mag[i] < -25:  # just an idea to discard peak searching during silence (or low noise..)
                peak_loc[i] = peak_loc[i - 1]  # silence : don't change pitch interpretation
        else:
            if peak_mag[i] < -35:  # We allow harmonics to be 10dB lower than the main peak
                peak_loc[i] = peak_loc[i - 1]

    return peak_loc, peak_mag


# Erase a trajectory of xdb by setting the frequency bin around the input trajectory to -80dB.
def flattenMaxPeak(xdB, maxPeakLoc):
    n_frames = xdB.shape[1]
    for frameIndex in range(n_frames):

        minfreq = int(maxPeakLoc[frameIndex]) - int(maxPeakLoc[frameIndex] / 16)
        maxfreq = int(maxPeakLoc[frameIndex]) + int(maxPeakLoc[frameIndex] / 16)

        for freqIndex in range(minfreq, maxfreq):
            xdB[freqIndex, frameIndex] = -80
    return xdB


# Median filter on the input segment sig
def movingMedian(sig, windowLength=5):
    sigSmooth = np.zeros(len(sig))
    for i in range(len(sig)):
        if i < math.floor(windowLength / 2) or i > len(sig) - math.floor(windowLength / 2) - 1:
            sigSmooth[i] = sig[i]
        else:
            sigSmooth[i] = np.median(sig[i - math.floor(windowLength / 2):i + math.floor(windowLength / 2)])
    return sigSmooth
