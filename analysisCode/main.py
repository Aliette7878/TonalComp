import numpy as np
import librosa.display
import math
import matplotlib.pyplot as plt
import glob
import librosa.display
import wave
import struct
import scipy.signal

from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm


demo_files = []
for file in glob.glob("..\\demo_sound\\*.wav"):
    demo_files.append(file)

print(demo_files)

# example_number = int(input(".wav example number = "));
example_number = 3
path_name = demo_files[example_number - 1]
audio, Fs = librosa.load(path_name, sr=None)
print("Opening " + path_name)
print("Fs: ", Fs)

# ------------------------------------------ USER SETTINGS ------------------------------------------

ParabolicInterpolation = True
MissingFundSearch = False  # # Do you want to look for a missing fundamental ?
PhaseConsidering = True  # Maybe not a user option
deletingShortTracks = 1  # If deleting short trajectories

# Bandwidth
f_low = 100  # will limit d_f, strongly impact the final sound
f_high = 18000  # can not be higher than 19 000 Hz

# Possibility to over-write N_fft, and Win_length

# Number of peaks to look for the fundamental
numberOfPeaks = 4  # 2 for the flute, 4 for the harmonica..
if MissingFundSearch:
    numberOfPeaks = 4  # We need a good average of the gap between each peaks to estimate the fundamental

N_moving_median = 19  # Maybe not a user option

# ------------------------------------------ WINDOWING ------------------------------------------

# Number of harmonics (including the fundamental)
N_h = 12

# Window type
Win_type = "hamming"
# Smoothness factor
L = 4

# Frequency separation
d_fmin = 20  # DO NOT CHANGE - The frequency resolution is limited, to balance with the temporal resolution
d_f = f_low
if f_low < d_fmin:
    d_f = d_fmin
    print('\033[93m' + f"WARNING: f_low chosen too low, and therefore changed to {d_fmin}" + '\033[0m')

# Window length
# Depends on our sampling frequency and the desired frequency separation
Win_length = math.floor(L * Fs / d_f)  # is higher when d_f is low

# Number of FFT samples
# N_fft should be at least the window's length, and should respect the JND criteria
N_fft = np.max([2 ** math.ceil(math.log2(Win_length)), 2 ** math.ceil(math.log2(Fs / (2 * 3)))])

# Nyquist index
Nyq = math.floor(N_fft / 2)

# Main lobe width
MainLobe = math.floor(L * N_fft / Win_length)

# Hop ratio
Hop_ratio = 4

# Hop length
Hop_length = int(Win_length / Hop_ratio)

# Number of frames
n_frames = math.floor((len(audio)) / Hop_length) + 1

# Minimum duration of a trajectory in seconds
minTrajDuration_seconds = 0.2
minTrajDuration = round(minTrajDuration_seconds * Fs / Hop_length) #in frames

# Define the array of amplitudes for the N_h harmonics
if N_h == 1:
    Amplitudes_array = 1
else:
    Amplitudes_array = np.ones(N_h) * 0.5
    Amplitudes_array[0] = 1

if __name__ == "__main__":  # This prevents the execution of the following code if main.py is imported in another script
    print("Peak resolution d_f = " + str(d_f) + " Hz")
    print("N_fft = " + str(N_fft))
    print("Window length = " + str(Win_length))
    print("Hop length = " + str(Hop_length))
    print("n frames = " + str(n_frames))
    print("Audio length = " + str(len(audio)))
    print("Min trajectory duration = " + str(minTrajDuration_seconds) + " seconds, " + str(minTrajDuration) + " frames")
    print("Amplitudes array = " + str(Amplitudes_array))

# ------------------------------------------ STFT ------------------------------------------

if __name__ == "__main__":  # This prevents the execution of the following code if main.py is imported in another script

    X_complex = librosa.stft(
        audio,
        win_length=Win_length,
        window=Win_type,
        n_fft=N_fft,
        hop_length=Hop_length, )

    X = np.abs(X_complex)
    # X_phase = np.arctan(X_complex.imag / X_complex.real)
    X_db = librosa.amplitude_to_db(X, ref=np.max)

    # Frequency spectrogram

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(X_db, y_axis='log', x_axis='time', sr=Fs)

    plt.colorbar(format='%+2.0f dB')
    plt.title('Frequency spectrogram')
    plt.tight_layout()
    plt.show()


# ------------------------------------------ PEAK TRACKING ------------------------------------------

# peakFinding goes through all frames of xdB. For each frame, it keeps the loudest frequency's localization and magnitude.
# If the peak's magnitude is too low (<-25 or -35), it is interpreted as a silence. The pitch is considered as the same as the precedent frame, but silent.
def peakFinding(xdB):
    n_frames_pf = xdB.shape[1]
    peak_loc = np.zeros(n_frames_pf)
    peak_mag = np.zeros(n_frames_pf)

    for i in range(n_frames_pf):
        peak_loc[i] = np.argmax(xdB[:, i])
        peak_mag[i] = xdB[int(peak_loc[i]), i]  # np.argmax return float even though here it's always int.

    return peak_loc, peak_mag


# Erase a trajectory of xdb by setting the frequency bin around the input trajectory to -80dB.
def flattenMaxPeak(xdB, maxPeakLoc):
    n_frames_fmp = xdB.shape[1]
    for frameIndex in range(n_frames_fmp):

        minfreq = int(maxPeakLoc[frameIndex]) - int(maxPeakLoc[frameIndex] / 16)
        maxfreq = min(xdB.shape[0] - 1, int(maxPeakLoc[frameIndex]) + int(maxPeakLoc[frameIndex] / 16))

        for freqIndex in range(minfreq, maxfreq):
            xdB[freqIndex, frameIndex] = -80
    return xdB


def computeAllPeaks(Xdb, numOfPeaks, iToFreq):

    peakLocList = []  # Too constraining to initialise with numpy
    peakMagList = []
    X_db_actualStep = Xdb.copy()
    X_db_actualStep[0:int(f_low / iToFreq), :] = np.min(
        Xdb)  # To avoid looking for sounds under the lowest sound we want to hear

    # for each trajectory : find the maximum trajectory, save it and erase it in xdb.
    for j in range(numOfPeaks):
        Peak_loc, Peak_mag = peakFinding(X_db_actualStep)
        peakLocList.append(Peak_loc)
        peakMagList.append(Peak_mag)
        X_db_actualStep = flattenMaxPeak(X_db_actualStep, Peak_loc)

    peakLocList = np.array(peakLocList)
    peakMagList = np.array(peakMagList)

    return peakLocList, peakMagList


if __name__ == "__main__":  # This prevents the execution of the following code if main.py is imported in another script

    indexToFreq = Fs / N_fft
    peakLoc_List, peakMag_List = computeAllPeaks(X_db, numberOfPeaks, indexToFreq)

    fundThroughFrame = np.amin(peakLoc_List, axis=0)
    fundThroughFrameSmoother = scipy.signal.medfilt(fundThroughFrame, N_moving_median)

    # Plotting results
    plt.figure(figsize=(15, 8))
    plt.subplot(311)

    symbolList = ['o', 'x', 'v', '*', 'h', '+', 'd', '^']
    legendList = []

    for j in range(numberOfPeaks):
        pkloc = peakLoc_List[j]
        plt.plot(np.arange(len(pkloc)), indexToFreq * pkloc, symbolList[j % len(symbolList)])
        legendList.append("peak " + str(j + 1))

    plt.legend(legendList)

    plt.subplot(312)
    plt.plot(np.arange(len(fundThroughFrame)), indexToFreq * fundThroughFrame)

    plt.subplot(313)
    plt.plot(np.arange(len(fundThroughFrameSmoother)), indexToFreq * fundThroughFrameSmoother)

    plt.show()


# ------------------------------------------ FIND HARMONICS WITH BLOCKS METHOD ------------------------------------------

# The parabola is given by y(x) = a*(x-p)²+b where y(-1) = alpha, y(0) = beta, y(1) = gamma
def parabolic_interpolation(alpha, beta, gamma):
    location = 0
    value = beta
    if alpha - 2 * beta + gamma != 0:
        location = 0.5 * (alpha - gamma) / (alpha - 2 * beta + gamma)
        value = beta - location * (alpha - gamma) / 4
    return [value, location]


# Check if peaks are spaced by the computed fundamental, or by one of its divisors
def real_fundamental(peak_freqs, foo):
    peak_freqs2 = peak_freqs.copy()
    peak_freqs2.sort()
    gap_list = [np.abs(peak_freqs2[0:len(peak_freqs2) - 1] - peak_freqs2[1:len(peak_freqs2)])]
    gap_list = np.array(gap_list)
    gap_list = gap_list[gap_list > 0.9 * f_low / indexToFreq]  # discard gaps under f_low
    divisor = 1
    if gap_list.size > 0:
        gap = np.min(gap_list)
        while np.abs(foo / divisor - gap) > np.abs(foo / (divisor + 1) - gap) and divisor < 10:
            divisor = divisor + 1
    return divisor


def findHarmonics_blockMethod(xdB, fundamentalList, peakLocList, indexToFreq, missingFundSearch):
    n_frames_fhbm = xdB.shape[1]
    # Width of the research block
    Bw = 2 * MainLobe

    # Initialization of storage vectors (which INCLUDE the fundamtental)
    harmonic_db = np.zeros((n_frames_fhbm, N_h))
    harmonic_freq = np.zeros((n_frames_fhbm, N_h))

    # Silence threshold
    silence_thres = 0.9 * np.min(xdB)

    if missingFundSearch:
        Divisor = []
        for n in range(n_frames_fhbm):
            divisor = real_fundamental(peakLocList[:, n], fundamentalList[n])
            Divisor.append(divisor)
        DivisorSmoother = scipy.signal.medfilt(Divisor, N_moving_median)

    # Building Harmonic_db and Harmonic_freq
    for n in range(n_frames_fhbm):
        Bw = int(0.9 * fundamentalList[n] / indexToFreq)

        if missingFundSearch:
            div = DivisorSmoother[n]
            fo = fundamentalList[n] / div
        else:
            fo = fundamentalList[n]
            div = 1

        for h in range(N_h):

            # Theoretical harmonic frequency
            k_th = math.floor((h + div) * fo)

            # If the theoretical harmonic frequency is in the bandwidth, we can apply the block method
            if k_th * indexToFreq > np.min([f_high, 0.90 * Fs / 2]):
                harmonic_db[n, h] = np.min(xdB)
                if n > 1:
                    harmonic_freq[n, h] = harmonic_freq[n - 1, h]
                else:
                    harmonic_freq[n, h] = 0

            else:
                # Draw the research block
                k_inf = max(0, k_th - Bw)
                k_inf = min(k_inf, Nyq)
                Block = xdB[k_inf:min(k_inf + 2 * Bw - 1, Nyq), n]

                maxB = max(Block)
                k_maxB = np.argmax(Block, axis=0)

                # Interpolation
                if 0 < k_maxB < 2 * (Bw - 1) and ParabolicInterpolation:
                    alpha = Block[k_maxB - 1]
                    beta = Block[k_maxB]
                    gamma = Block[k_maxB + 1]
                    [int_mag, int_loc] = parabolic_interpolation(alpha, beta, gamma)
                    int_mag = min(int_mag, 0)
                else:
                    [int_mag, int_loc] = [maxB, k_maxB]
                int_loc = k_inf + k_maxB + int_loc

                # Store the interpolated peak
                harmonic_db[n, h] = int_mag
                if int_mag < silence_thres and n > 0:  # the pitch remains the same if a silence is detected
                    harmonic_freq[n, h] = harmonic_freq[n - 1, h]
                else:
                    harmonic_freq[n, h] = indexToFreq * int_loc

    return harmonic_freq, harmonic_db


# Smoothing the harmonics trajectories
def smootherHarmonics(harmonic_freq, NmovingMedian):
    harmonic_freqSmoother = harmonic_freq
    for harmo in range(N_h):
        harmonic_freqSmoother[:, harmo] = scipy.signal.medfilt(harmonic_freq[:, harmo], NmovingMedian)
    return harmonic_freqSmoother


def plotHarmoIntensity(HarmonicFreqSmoother, HarmonicDB):

    fig, axs = plt.subplots(1, 1, sharex=True, sharey=True)

    x = np.arange(len(HarmonicFreqSmoother[:, 0]))
    y = HarmonicFreqSmoother
    y = y.reshape(y.size)

    dydx = HarmonicDB
    norm = plt.Normalize(np.min(dydx), np.max(dydx))

    for h in range(N_h):
        y = HarmonicFreqSmoother[:, h]
        dydx = HarmonicDB[:, h]

        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        lc = LineCollection(segments, cmap='Greys', norm=norm)
        lc.set_array(dydx)
        lc.set_linewidth(2)
        line = axs.add_collection(lc)
        # fig.colorbar(line, ax=axs)

    axs.set_xlim(x.min(), x.max())
    axs.set_ylim(np.min(HarmonicFreqSmoother), np.max(HarmonicFreqSmoother))
    plt.title('Smoothed fundamental and its harmonics - colored intensity')
    plt.xlabel("Frames")
    plt.ylabel("Hz")
    plt.show()


# Compute and plot the harmonics
if __name__ == "__main__":
    Harmonic_freq, Harmonic_db = findHarmonics_blockMethod(X_db, fundThroughFrameSmoother, peakLoc_List, indexToFreq,
                                                           MissingFundSearch)
    Harmonic_freqSmoother = smootherHarmonics(Harmonic_freq, N_moving_median)

    plt.figure(figsize=(15, 8))

    plt.subplot(211)
    plt.title('Fundamental and its harmonics - block research method')
    plt.plot(np.arange(len(Harmonic_freq)), Harmonic_freq)
    plt.ylabel("Hz")
    plt.xlabel("Frames")

    plt.subplot(212)
    plt.title('Smoothed fundamental and its harmonics - block research method')
    plt.plot(np.arange(len(Harmonic_freqSmoother)), Harmonic_freqSmoother)
    plt.xlabel("Frames")
    plt.ylabel("Hz")
    plt.tight_layout()

    plt.show()

    # 2nd plot window: plotting harmonics with colored intensity
    plotHarmoIntensity(Harmonic_freqSmoother, Harmonic_db)

# ------------------------------------------- TRAJECTORIES ------------------------------------------

#We assume that we have a representation that is [fundamental, 1st harmonic, 2nd harm...] through all frames
#Even in the moments of silence, there are N_h harmonics, just with very low amplitude -> those might get set to zero if too fast


def build_trajectories(Harm_db, Harm_freq):

    traj = np.zeros((Harm_db.shape[0], Harm_db.shape[1]))
    traj_freq = np.zeros((Harm_db.shape[0], 2 * Harm_db.shape[1]))
    traj_db = np.zeros((Harm_db.shape[0], 2 * Harm_db.shape[1]))

    for m in range(Harm_db.shape[0] - 1):

        for i in range(Harm_db.shape[1]):

            if m == 0:
                traj[m, i] = 1
                traj_freq[m, i * 2] = Harm_freq[m, i]
                traj_db[m, i * 2] = Harm_db[m, i]
            else:
                freq_distance = abs(Harm_freq[m, i] - Harm_freq[m - 1, i])  # measure freq distance
                freq_dev_offset = (Harm_freq[m - 1, i]) * (pow(2, 1 / 12) - 1) * 0.4     # a bit lower than half of a half-tone
                if freq_distance < freq_dev_offset:     # if belongs to the same trajectory
                    traj[m, i] = traj[m - 1, i]
                    if traj_freq[m - 1, i * 2] == 0:
                        traj_freq[m, i * 2 + 1] = Harm_freq[m, i]
                        traj_db[m, i * 2 + 1] = Harm_db[m, i]
                    else:
                        traj_freq[m, i * 2] = Harm_freq[m, i]
                        traj_db[m, i * 2] = Harm_db[m, i]
                else:                                    # if new trajectory to begin
                    if traj[m - 1, i] == 1:
                        traj[m, i] = 2
                    elif traj[m - 1, i] == 2:
                        traj[m, i] = 1
                    if traj_freq[m - 1, i * 2] == 0:
                        traj_freq[m, i * 2] = Harm_freq[m, i]
                        traj_db[m, i * 2] = Harm_db[m, i]
                    else:
                        traj_freq[m, i * 2 + 1] = Harm_freq[m, i]
                        traj_db[m, i * 2 + 1] = Harm_db[m, i]

    return traj, traj_freq, traj_db


def delete_short_trajectories(traj, traj_freq, traj_db, Harm_db, min_traj_duration, min_amp_db):

    Harm_db_filtered = np.copy(Harm_db)

    for i in range(traj.shape[1]):

        for m in range(traj.shape[0] - 1):

            if m == 0:
                traj_start = m
            elif traj[m, i] != traj[m - 1, i]:
                traj_start = m
            if traj[m, i] != traj[m + 1, i]:
                traj_end = m
                if (traj_end - traj_start >= 0) & (traj_end - traj_start < min_traj_duration):
                    traj[traj_start: traj_end + 1, i] = 0
                    traj_freq[traj_start: traj_end + 1, i * 2: i * 2 + 2] = 0
                    traj_db[traj_start: traj_end + 1, i * 2: i * 2 + 2] = min_amp_db
                    Harm_db_filtered[traj_start: traj_end + 1, i] = min_amp_db

    return traj, traj_freq, traj_db, Harm_db_filtered


def smooth_trajectories_freq(traj, traj_freq, Harm_freq, min_traj_duration):

    Harm_freq_filtered = np.copy(Harm_freq)

    for i in range(traj.shape[1]):

        first_start_found = 0

        for m in range(traj.shape[0] - 1):

            if (m == 0) & (traj[m, i] != 0):
                traj_start = m
                first_start_found = 1
            elif (traj[m, i] != 0) & (traj[m, i] != traj[m - 1, i]):
                traj_start = m
                first_start_found = 1
            if first_start_found & (traj[m, i] != traj[m + 1, i]):
                traj_end = m
                if (traj_end - traj_start >= 0) & (traj_end - traj_start > min_traj_duration):

                    # kernel = (traj_end - traj_start) if ((traj_end - traj_start) % 2) else (traj_end - traj_start - 1)
                    # if (traj_end - traj_start) < 9:
                    #    kernel = (traj_end - traj_start) if ((traj_end - traj_start) % 2) else (traj_end - traj_start - 1)
                    # else:
                    #    kernel = 9
                    Harm_freq_filtered[traj_start: traj_end + 1, i] = np.median(Harm_freq[traj_start: traj_end + 1, i])
                    #scipy.signal.medfilt(Harm_freq[traj_start: traj_end + 1, i], kernel)
                    if traj[traj_end, i] == 1:
                        traj_freq[traj_start: traj_end + 1, 2 * i] =\
                            np.median(traj_freq[traj_start: traj_end + 1, 2 * i])
                    elif traj[traj_end, i] == 2:
                        traj_freq[traj_start: traj_end + 1, 2 * i + 1] =\
                            np.median(traj_freq[traj_start: traj_end + 1, 2 * i + 1])
                    #scipy.signal.medfilt(traj_freq[traj_start: traj_end + 1, i], kernel)

    return traj, traj_freq, Harm_freq_filtered


def plotSmoothTrajIntensity(trajectoriesFreq, trajectoriesDB, miny, maxy):

    # Plot the smoothened trajectories, with intensity
    fig, axs = plt.subplots(1, 1, sharex=True, sharey=True)

    x = np.arange(len(trajectoriesFreq[:, 0]))
    y = trajectoriesFreq
    y = y.reshape(y.size)

    dydx = trajectoriesDB
    norm = plt.Normalize(np.min(dydx), np.max(dydx))

    for h in range(N_h*2):
        y = trajectoriesFreq[:, h]
        y[y<=0] = np.nan
        dydx = trajectoriesDB[:, h]

        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        lc = LineCollection(segments, cmap='Blues', norm=norm)
        lc.set_array(dydx)
        lc.set_linewidth(2)
        line = axs.add_collection(lc)

    axs.set_xlim(x.min(), x.max())
    axs.set_ylim(miny, maxy)
    plt.title('Smoothed fundamental and its harmonics - after filtering trajectories')
    plt.xlabel("Frames")
    plt.ylabel("Hz")
    plt.show()


if __name__ == "__main__":

    minAmp_db = np.min(Harmonic_db)

    # Computing trajectories
    trajectories, trajectories_freq, trajectories_db = build_trajectories(Harmonic_db, Harmonic_freqSmoother)

    # Plot the trajectories, without representing intensity
    if trajectories_freq.shape[1] > 0:

        y = np.copy(trajectories_freq)
        y[y <= 0] = np.nan
        plt.plot(np.arange(len(y)), y, 'k')
        plt.xlabel('Frames')
        plt.ylabel('Frequency (Hz)')
        plt.title('Trajectories')

    plt.tight_layout()
    plt.show()

    # Deleting short trajectories (below specific minimum duration)
    trajectories, trajectories_freq, trajectories_db, Harmonic_db_filtered =\
        delete_short_trajectories(trajectories, trajectories_freq, trajectories_db, Harmonic_db, minTrajDuration
                                  , minAmp_db)

    # Smoothening out frequency variation within each trajectory
    trajectories, trajectories_freq, Harmonic_freq_filtered =\
        smooth_trajectories_freq(trajectories, trajectories_freq, Harmonic_freqSmoother, minTrajDuration)

    # Plot the smoothened trajectories, with intensity
    min_y, max_y = np.min(Harmonic_freq), np.max(Harmonic_freq)
    plotSmoothTrajIntensity(trajectories_freq, trajectories_db, min_y, max_y)









# ------------------------------------------ PART 2 : SYNTHESIS ------------------------------------------

def linear_interpolation(a, b, n):
    s = np.arange(n) * (b - a) / (n - 1) + np.ones(n) * a
    return s


def oscillators_bank_synthesis(harm_db, harm_freq, f_s, hop_length, filtering_tracks):

    num_frames = harm_db.shape[0]
    n_h = harm_db.shape[1]

    # Time axis
    time = np.arange(0, hop_length * num_frames)

    # db to amplitude
    harm_amp = librosa.db_to_amplitude(harm_db)  # deleted the part "np.max(X)"

    # Allocate the output vector
    out_bankosc = np.zeros(num_frames * hop_length)

    for i in range(n_h):

        # Generate the interpolated amp, freq and phase, between each frame

        oscillator = np.zeros(num_frames * hop_length)

        IntAmp = np.zeros(num_frames * hop_length)
        IntPhase = np.zeros(num_frames * hop_length)
        IntFreq = np.zeros(num_frames * hop_length)

        for m in range(num_frames - 1):
            IntAmp[m * hop_length:(m + 1) * hop_length] = linear_interpolation(harm_amp[m, i], harm_amp[m + 1, i],
                                                                               hop_length)

            IntFreq[m * hop_length:(m + 1) * hop_length] = np.ones(hop_length) * harm_freq[m, i]

            # IntPhase[m * hop_length:(m + 1) * hop_length] = linear_interpolation(fundPhase[m], fundPhase[m + 1], hop_length)

        oscillator = IntAmp * np.sin(2 * np.pi * IntFreq * time / f_s)
        # bad vibrato doesn't come from the amplitude, but from IntPhase.
        # It works better without the phase

        out_bankosc = out_bankosc + oscillator

    # Normalizing out
    out_bankosc = out_bankosc * (np.max(audio) - np.min(audio)) / (np.max(out_bankosc) - np.min(out_bankosc))
    return out_bankosc

# ---------------------------------------- ADDITIVE SYNTHESIS: STARTING FROM ONLY FUNDAMENTAL ------------------------------------------

def oscillators_bank_synthesis_additive(harm_db, harm_freq, f_s, hop_length, amplitude_array):

    num_frames = harm_db.shape[0]
    n_h = harm_db.shape[1]

    # Time axis
    time = np.arange(0, hop_length * num_frames)

    # Fundamental: db to amplitude
    harm_amp = librosa.db_to_amplitude(harm_db)  # deleted the part "np.max(X)"
    fund_amp = harm_amp[:, 0]

    # Fundamental: frequency
    fund_freq = harm_freq[:, 0]

    # Allocate the output vector
    out_bankosc = np.zeros(num_frames * hop_length)

    for i in range(n_h):

        # Generate the interpolated amp, freq and phase, between each frame

        oscillator = np.zeros(num_frames * hop_length)

        IntAmp = np.zeros(num_frames * hop_length)
        IntPhase = np.zeros(num_frames * hop_length)
        IntFreq = np.zeros(num_frames * hop_length)

        for m in range(num_frames - 1):
            IntAmp[m * hop_length:(m + 1) * hop_length] = linear_interpolation(fund_amp[m], fund_amp[m + 1],
                                                                               hop_length) * amplitude_array[i]

            IntFreq[m * hop_length:(m + 1) * hop_length] = np.ones(hop_length) * fund_freq[m] * (i + 1)

            # IntPhase[m * hop_length:(m + 1) * hop_length] = linear_interpolation(fundPhase[m], fundPhase[m + 1], hop_length)

        oscillator = IntAmp * np.sin(2 * np.pi * IntFreq * time / f_s)

        out_bankosc = out_bankosc + oscillator

    # Normalizing out
    out_bankosc = out_bankosc * (np.max(audio) - np.min(audio)) / (np.max(out_bankosc) - np.min(out_bankosc))
    return out_bankosc


# ------------------------------------------ SOUND - WAV FILE CREATION ------------------------------------------

def wave_file_creation(ex_number, out_bankosc, f_s, type_additive):
    # Open the wav file
    file_name = "Synthesized_Osc_" + ("additive_" if type_additive else "") + "example_" + str(example_number) + ".wav"
    wav_file = wave.open(file_name, "w")
    print("Saving " + file_name + "...")

    # Writing parameters
    data_size = len(out_bankosc)
    amp = 64000.0  # multiplier for amplitude
    nchannels = 1
    sampwidth = 2  # 2 for stereo
    comptype = "NONE"
    compname = "not compressed"

    # Set writing parameters
    wav_file.setparams((nchannels, sampwidth, Fs, data_size, comptype, compname))

    # Write out in the wav file
    for s in out_bankosc:
        wav_file.writeframes(struct.pack('h', int(s * amp / 2)))

    wav_file.close()
    print(file_name + " saved")


if __name__ == "__main__":  # For now the whole synthesis part doesn't exists outside of main

    # 1. Part with tracking the original harmonics

    if deletingShortTracks:
        bankosc = oscillators_bank_synthesis(Harmonic_db_filtered, Harmonic_freq_filtered, Fs, Hop_length, deletingShortTracks)
    else:
        bankosc = oscillators_bank_synthesis(Harmonic_db, Harmonic_freqSmoother, Fs, Hop_length, deletingShortTracks)

    # Writing the file with original harmonics
    wave_file_creation(example_number, bankosc, Fs, 0)

    # 2. Part with controlling harmonics

    bankosc_add = oscillators_bank_synthesis_additive(Harmonic_db_filtered, Harmonic_freq_filtered, Fs, Hop_length, Amplitudes_array)

    # Writing the file with controlled harmonics
    wave_file_creation(example_number, bankosc_add, Fs, 1)
