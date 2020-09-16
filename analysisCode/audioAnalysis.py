import librosa.display
import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import tkinter as tk

from main import findPeaksScipy, plot_fundamental, findHarmonics_blockMethod, smootherHarmonics, \
    plot_harmonics, build_trajectories, delete_short_trajectories, smooth_trajectories_freq, \
    plot_trajectories, plot_harmonics_intensity, plot_synthesis, \
    wave_file_creation, resynthesis, custom_synthesis, N_moving_median, minTrajDuration, sustain_amp


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
        self.Harmonic_freqSmoother = None
        self.Harmonic_db = None
        self.Harmonic_db_filtered = None
        self.Harmonic_freqMedian = None
        self.trajectories = None

    def showframe(self, frameIndex):
        fig2 = plt.figure(figsize=(20, 15))
        plt.plot(self.indexToFreq * np.arange(self.X_db[:1500, frameIndex].size), self.X_db[:1500, frameIndex])
        fig2.show()

    def showfundamental(self):

        self.fundThroughFrame = np.zeros(self.n_frames)

        for m in range(self.n_frames):
            self.fundThroughFrame[m] = findPeaksScipy(self.X_db[:, m], self.analysisParams.f_low / self.indexToFreq)[0]

        self.fundThroughFrameSmoother = scipy.signal.medfilt(self.fundThroughFrame, N_moving_median)

        plot_fundamental(self.fundThroughFrame * self.indexToFreq, self.fundThroughFrameSmoother * self.indexToFreq)

    def show_harmonics(self):

        # compute fundamental with the dedicated function implemented in main.py
        if self.fundThroughFrameSmoother is None:
            self.fundThroughFrame = np.zeros(self.n_frames)
            for m in range(self.n_frames):
                self.fundThroughFrame[m] = \
                    findPeaksScipy(self.X_db[:, m], self.analysisParams.f_low / self.indexToFreq)[0]
            self.fundThroughFrameSmoother = scipy.signal.medfilt(self.fundThroughFrame, N_moving_median)

        # compute harmonics
        Harmonic_freq, self.Harmonic_db = findHarmonics_blockMethod(self.X_db, self.fundThroughFrameSmoother,
                                                                    self.indexToFreq, False)
        self.Harmonic_freqSmoother = smootherHarmonics(Harmonic_freq, N_moving_median)

        plot_harmonics(Harmonic_freq, self.Harmonic_freqSmoother)

    def show_trajectories(self):

        # compute fundamental with the dedicated function implemented in main.py
        if self.fundThroughFrameSmoother is None:
            self.fundThroughFrame = np.zeros(self.n_frames)
            for m in range(self.n_frames):
                self.fundThroughFrame[m] = findPeaksScipy(self.X_db[:, m], self.analysisParams.f_low / self.indexToFreq)[0]
            self.fundThroughFrameSmoother = scipy.signal.medfilt(self.fundThroughFrame, N_moving_median)

        # compute harmonics
        Harmonic_freq, self.Harmonic_db = findHarmonics_blockMethod(self.X_db, self.fundThroughFrameSmoother, self.indexToFreq, False)
        self.Harmonic_freqSmoother = smootherHarmonics(Harmonic_freq, N_moving_median)

        minAmp_db = np.min(self.Harmonic_db)

        # Computing trajectories
        self.trajectories, trajectories_freq, trajectories_db = build_trajectories(self.Harmonic_db,
                                                                                   self.Harmonic_freqSmoother)

        # Plot the found trajectories
        plot_trajectories(trajectories_freq)

        # Deleting short trajectories (below specific minimum duration)
        self.trajectories, trajectories_freq, trajectories_db, self.Harmonic_db_filtered = \
            delete_short_trajectories(self.trajectories, trajectories_freq, trajectories_db, self.Harmonic_db,
                                      minTrajDuration, minAmp_db)

        # Plot the smoothened trajectories, with intensity
        min_y, max_y = np.min(Harmonic_freq), np.max(Harmonic_freq)
        title_traj = "Fundamental and its harmonics after deleting short trajectories"

        plot_harmonics_intensity(self.Harmonic_freqSmoother, self.Harmonic_db, trajectories_freq, trajectories_db,
                                 title_traj, min_y, max_y)

        # Smoothening out frequency variation within each trajectory with Median
        self.trajectories, trajectories_freq, self.Harmonic_freqMedian = \
            smooth_trajectories_freq(self.trajectories, trajectories_freq, self.Harmonic_freqSmoother, minTrajDuration)

        # Harmonic_freqSmoother should be used for re-synthesis
        # Harmonic_freq_Median should be used for custom_synthesis

    def resynthesize(self):
        if self.Harmonic_db_filtered is None:
            self.show_trajectories()
        bankosc_resynth = resynthesis(self.Harmonic_db_filtered, self.Harmonic_freqSmoother, self.Fs, self.hop_length)

        # Plotting the original and the re-synthesised audio files in time domain
        plot_synthesis(self.audio, bankosc_resynth, "Re-synthesized audio file")

        f = tk.filedialog.asksaveasfile(initialfile="resynthesisedSound", mode='w', defaultextension=".wav")
        path_name = f.name
        f.close()
        print(path_name)

        # Writing the file with controlled harmonics
        wave_file_creation(bankosc_resynth, self.Fs, path_name)

    def customSynth(self, amplitude_array, adsrBool, inharmonicity_array, attack, decay):
        if self.Harmonic_freqMedian is None:
            self.show_trajectories()
        bankosc_custom_synth = custom_synthesis(self.Harmonic_db_filtered, self.Harmonic_freqMedian, self.Harmonic_db,
                                                self.Harmonic_freqSmoother, self.trajectories, amplitude_array, inharmonicity_array,
                                                attack, decay, sustain_amp, self.Fs, self.hop_length, adsrBool)

        # Plotting the original and customarily synthesised audio files in time domain
        plot_synthesis(self.audio, bankosc_custom_synth, "Customarily synthesized audio file")
        # Writing the file with controlled harmonics
        f = tk.filedialog.asksaveasfile(initialfile="customSynthesisedSound", mode='w', defaultextension=".wav")
        path_name = f.name
        f.close()
        print(path_name)
        wave_file_creation(bankosc_custom_synth, self.Fs, path_name)


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
