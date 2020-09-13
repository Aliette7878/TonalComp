import librosa.display
import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal

from main import computeAllPeaks, findHarmonics_blockMethod, smootherHarmonics, build_trajectories, \
    delete_short_trajectories, smooth_trajectories_freq, plotSmoothTrajIntensity, N_moving_median, numberOfPeaks, minTrajDuration, findPeaksScipy


class AudioAnalysis:

    def __init__(self, pathName, analysisParams, winLength_mul=1, nfft_mul=1):
        # self.analysisParams = analysisParams      #for the moment not needed outside
        print("Opening " + pathName)
        self.pathName = pathName
        self.audio, self.Fs = librosa.load(pathName, sr=None)       # mp3 not supported yet (can be with ffmpeg)
        print("Fs: ", self.Fs)
        self.win_length = math.floor(analysisParams.L * self.Fs / analysisParams.d_f)
        self.win_length = math.floor(self.win_length*winLength_mul)
        # N_fft should be at least the window's length, and should respect the JND criteria
        self.N_fft = max(2 ** math.ceil(math.log2(self.win_length)), 2 ** math.ceil(math.log2(self.Fs / 3)))
        self.N_fft = math.floor(self.N_fft**nfft_mul)
        self.indexToFreq = self.Fs / self.N_fft
        self.hop_length = math.floor(self.win_length / analysisParams.hop_ratio)

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

    def showpeaks(self, numbOfPeaks):

        peakLoc_List, peakMag_List = computeAllPeaks(self.X_db, numbOfPeaks, self.indexToFreq)

        fundThroughFrame = np.amin(peakLoc_List, axis=0)    # Depending on the number of peaks computed, this can be
        # irrelevant (too much peak = peaks under fundamental, only 1 or 2 peaks can catch harmonics only)
        fundThroughFrameSmoother = scipy.signal.medfilt(fundThroughFrame, N_moving_median)

        plt.figure(figsize=(15, 8))

        plt.subplot(311)
        symbolList = ['o', 'x', 'v', '*', 'h', '+', 'd', '^']
        legendList = []
        for j in range(numbOfPeaks):
            pkloc = peakLoc_List[j]
            plt.plot(np.arange(len(pkloc)), self.indexToFreq * pkloc, symbolList[j % len(symbolList)])
            legendList.append("peak " + str(j + 1))
        plt.legend(legendList)

        plt.subplot(312)
        plt.plot(np.arange(len(fundThroughFrame)), self.indexToFreq * fundThroughFrame)

        plt.subplot(313)
        plt.plot(np.arange(len(fundThroughFrameSmoother)), self.indexToFreq * fundThroughFrameSmoother)
        plt.show(block=False)

    def show_trajectories(self):

        # compute peaks with numberOfPeaks set in main.py
        peakLoc_List, peakMag_List = computeAllPeaks(self.X_db, numberOfPeaks, self.indexToFreq)

        fundThroughFrame = np.amin(peakLoc_List, axis=0)
        fundThroughFrameSmoother = scipy.signal.medfilt(fundThroughFrame, N_moving_median)


        #compute harmonics
        Harmonic_freq, Harmonic_db = findHarmonics_blockMethod(self.X_db, fundThroughFrameSmoother, peakLoc_List,
                                                        self.indexToFreq, False)    # No missing fund search for now
        Harmonic_freqSmoother = smootherHarmonics(Harmonic_freq, N_moving_median)


        # Computing trajectories
        trajectories, trajectories_freq, trajectories_db = build_trajectories(Harmonic_db, Harmonic_freqSmoother)

        minAmp_db = np.min(Harmonic_db)

        # Deleting short trajectories (below specific minimum duration)
        trajectories, trajectories_freq, trajectories_db, Harmonic_db_filtered = \
            delete_short_trajectories(trajectories, trajectories_freq, trajectories_db, Harmonic_db, minTrajDuration
                                      , minAmp_db)

        # Smoothening out frequency variation within each trajectory
        trajectories, trajectories_freq, Harmonic_freq_filtered = \
            smooth_trajectories_freq(trajectories, trajectories_freq, Harmonic_freqSmoother, minTrajDuration)

        # Plot the smoothened trajectories, with intensity
        min_y, max_y = np.min(Harmonic_freq), np.max(Harmonic_freq)
        plotSmoothTrajIntensity(trajectories_freq, trajectories_db, min_y, max_y)


class AnalysisParameters:

    d_fmin = 30
    hop_ratio = 4

    def __init__(self, fmin, fmax):
        self.f_low = fmin
        self.f_high = fmax
        self.N_h = 10
        self.win_type = "hamming"   # Window type
        self.L = 4                  # Smoothness factor of the chosen window's type
        self.d_f = self.f_low
        if self.f_low < self.d_fmin:
            self.d_f = self.d_fmin
            print('\033[93m' + f"WARNING: f_low chosen too low, and therefore changed to {self.d_fmin}")


