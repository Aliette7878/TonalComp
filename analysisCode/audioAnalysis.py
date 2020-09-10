import librosa.display
import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal

from main import peakFinding, flattenMaxPeak


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
        fundThroughFrameSmoother = scipy.signal.medfilt(fundThroughFrame, N_moving_median)

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


