import librosa.display
import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal

from main import findPeaksScipy, findHarmonics_blockMethod, smootherHarmonics, build_trajectories, \
    delete_short_trajectories, smooth_trajectories_freq, plotSmoothTrajIntensity, N_moving_median, minTrajDuration


class AudioAnalysis:

    def __init__(self, pathName, analysisParams, winLength_mul=1, nfft_mul=1):
        # self.analysisParams = analysisParams      #for the moment not needed outside
        print("Opening " + pathName)
        self.pathName = pathName
        self.audio, self.Fs = librosa.load(pathName, sr=None)       # mp3 not supported yet (can be with ffmpeg)
        print("Fs: ", self.Fs)
        self.analysisParams = analysisParams
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

        self.fundThroughFrame = None
        self.fundThroughFrameSmoother = None

    def showframe(self, frameIndex):
        fig2 = plt.figure(figsize=(20, 15))
        plt.plot(self.indexToFreq * np.arange(self.X_db[:1500, frameIndex].size), self.X_db[:1500, frameIndex])
        fig2.show()

    def showfundamental(self):

        self.fundThroughFrame = np.zeros(self.n_frames)

        for m in range(self.n_frames):
            self.fundThroughFrame[m] = findPeaksScipy(self.X_db[:, m], self.analysisParams.f_low / self.indexToFreq)[0]

        self.fundThroughFrameSmoother = scipy.signal.medfilt(self.fundThroughFrame, N_moving_median)

        plt.figure(figsize=(15, 8))

        plt.subplot(211)
        plt.plot(np.arange(len(self.fundThroughFrame)), self.indexToFreq * self.fundThroughFrame, '.')
        plt.title('Fundamental found with peak tracking')
        plt.xlabel("Frames")
        plt.ylabel('Hz')

        plt.subplot(212)
        plt.plot(np.arange(len(self.fundThroughFrameSmoother)), self.indexToFreq * self.fundThroughFrameSmoother, '.')
        plt.title('Smoothed fundamental')
        plt.xlabel("Frames")
        plt.ylabel('Hz')
        plt.show()

    def show_trajectories(self):

        # compute fundamental with the dedicated function implemented in main.py
        if self.fundThroughFrameSmoother is None:
            self.fundThroughFrame = np.zeros(self.n_frames)
            for m in range(self.n_frames):
                self.fundThroughFrame[m] = findPeaksScipy(self.X_db[:, m], self.analysisParams.f_low / self.indexToFreq)[0]
            self.fundThroughFrameSmoother = scipy.signal.medfilt(self.fundThroughFrame, N_moving_median)


        #compute harmonics
        Harmonic_freq, Harmonic_db = findHarmonics_blockMethod(self.X_db, self.fundThroughFrameSmoother, self.indexToFreq, False)
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

        # trajectories, trajectories_freq, Harmonic_freqMedian = \
        #     smooth_trajectories_freq(trajectories, trajectories_freq, Harmonic_freqSmoother, minTrajDuration)
        # # Harmonic_freqSmoother should be used for resynthesis
        # # Harmonic_freq_Median should be used for fundamental_synthesis


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


