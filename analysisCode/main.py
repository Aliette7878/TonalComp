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

example_number = 6
path_name = demo_files[example_number - 1]
audio, Fs = librosa.load(path_name, sr=None)
print("Opening " + path_name)
print("Fs: ", Fs)

# ------------------------------------------ USER SETTINGS ------------------------------------------

ParabolicInterpolation = True
MissingFundSearch = False  # # Do you want to look for a missing fundamental ?
envelopeADSR = False

# The bandwidth delimits the research of the fundamental and harmonics
f_low = 150
f_high = 18000


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

# Convert index of Nfft to frequency
indexToFreq = Fs / N_fft

# Minimum duration of a trajectory in seconds
minTrajDuration_seconds = 0.1
minTrajDuration = round(minTrajDuration_seconds * Fs / Hop_length) # in frames


# ----- Parameters of the custom synthesis ------

# ADSR envelope: attack and decay in seconds, sustain in amplitude
attack_sec = 0.2
decay_sec = 0.1
sustain_amp = 0.01

# Array of relative amplitudes for the N_h harmonics, with respect to the one of the fundamental frequency
# Amplitudes_array takes values from [0,1], where 1 corresponds to the original amplitude of the fundamental frequency
if N_h == 1:
    Amplitudes_array = [1]
else:
    Amplitudes_array = np.ones(N_h) * 0.5
    Amplitudes_array[0] = 1

# Array of relative frequency deviation for the N_h harmonics, with respect to exact multiples of the fundamental freq
# Inharmonicity_array takes values from [-1,1], where 0 corresponds to no deviation, and +-1 to +-quarter tone

if N_h == 1:
    Inharmonicity_array = [0]
else:
    Inharmonicity_array = np.zeros(N_h)
    Inharmonicity_array[1:N_h] = np.zeros(N_h - 1)


if __name__ == "__main__":  # This prevents the execution of the following code if main.py is imported in another script
    print("Peak resolution d_f = " + str(d_f) + " Hz")
    print("N_fft = " + str(N_fft))
    print("Window length = " + str(Win_length))
    print("Hop length = " + str(Hop_length))
    print("n frames = " + str(n_frames))
    print("Audio length = " + str(len(audio)))
    print("Min trajectory duration = " + str(minTrajDuration_seconds) + " seconds, " + str(minTrajDuration) + " frames")
    print("Amplitudes array = " + str(Amplitudes_array))
    print("Inharmonicity array = " + str(Inharmonicity_array))

# ------------------------------------------ STFT ------------------------------------------

def plot_stft(x_db, f_s):

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(x_db, y_axis='log', x_axis='time', sr=f_s)

    plt.colorbar(format='%+2.0f dB')
    plt.title('Frequency spectrogram')
    plt.xlabel('Time [s]')
    plt.ylabel('Frequency [Hz]')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":  # This prevents the execution of the following code if main.py is imported in another script

    X_complex = librosa.stft(
        audio,
        win_length=Win_length,
        window=Win_type,
        n_fft=N_fft,
        hop_length=Hop_length, )

    X = np.abs(X_complex)
    X_db = librosa.amplitude_to_db(X, ref=np.max)

    # Frequency spectrogram
    plot_stft(X_db, Fs)


# ------------------------------------------ PEAK TRACKING ------------------------------------------

def findPeaksScipy(X, threshold):
    x = np.copy(X)
    x[0:int(threshold)] = np.min(x)       # The peaks of low frequencies are not to be considered for the search

    height = max(np.max(x)-30, -70)      # The to-be-found peak of the fundamental is allowed to be 30dB under
                                         # the loudest harmonic, or not lower than -70dB

    distance = max(threshold, 1)         # The peaks are supposed to be spaced by at least a specified value, in this case f_low
    peakIndexes = scipy.signal.find_peaks(x, height=height, threshold=None, distance=distance)[0]
    if len(peakIndexes) == 0:
        peakIndexes = [0]
    return peakIndexes


def plot_fundamental(fund, fund_smooth):

    plt.figure(figsize=(15, 8))

    plt.subplot(211)
    plt.plot(np.arange(len(fund)), fund, '.')
    plt.title('Fundamental frequency found with peak tracking')
    plt.xlabel("Frames")
    plt.ylabel('Frequency [Hz]')

    plt.subplot(212)
    plt.plot(np.arange(len(fund_smooth)), fund_smooth, '.')
    plt.title('Fundamental frequency smoothed with a median filter')
    plt.xlabel("Frames")
    plt.ylabel('Frequency [Hz]')
    plt.show()


if __name__ == "__main__":  # This prevents the execution of the following code if main.py is imported in another script

    # SCIPY PART----------------

    fundThroughFrame = np.zeros(n_frames)

    for m in range(n_frames):
        fundThroughFrame[m] = findPeaksScipy(X_db[:, m], 0.9 * f_low/ indexToFreq)[0]

    fundThroughFrameSmoother = scipy.signal.medfilt(fundThroughFrame, N_moving_median)

    plot_fundamental(fundThroughFrame * indexToFreq, fundThroughFrameSmoother * indexToFreq)


# ------------------------------------------ FIND HARMONICS WITH BLOCKS METHOD -----------------------------------------


# The parabola is given by y(x) = a*(x-p)²+b where y(-1) = alpha, y(0) = beta, y(1) = gamma
def parabolic_interpolation(alpha, beta, gamma):
    location = 0
    value = beta
    if alpha - 2 * beta + gamma != 0:
        location = 0.5 * (alpha - gamma) / (alpha - 2 * beta + gamma)
        value = beta - location * (alpha - gamma) / 4
    return [value, location]


# Check if peaks are spaced by the computed fundamental, or by one of its divisors
def real_fundamental(peakLoc_List, foo):
    divisor = 1
    if len(peakLoc_List)>1:
        gap=peakLoc_List[1]-peakLoc_List[0]
        while np.abs(foo / divisor - gap) > np.abs(foo / (divisor + 1) - gap) and divisor < 10:
            divisor = divisor + 1
    return divisor


def findHarmonics_blockMethod(xdB, fundamentalList, indexToFreq, missingFundSearch):

    nframes = len(fundamentalList)
    # Initialization of storage vectors that include fundamental and harmonics
    harmonic_db = np.zeros((nframes, N_h))
    harmonic_freq = np.zeros((nframes, N_h))

    # Silence threshold
    silence_thres = 0.9 * np.min(xdB)

    if missingFundSearch:
        Divisor = []
        for n in range(nframes):
            peakLoc_List = findPeaksScipy(X_db[:, n], f_low / indexToFreq)
            divisor = real_fundamental(peakLoc_List, fundamentalList[n])
            Divisor.append(divisor)
        DivisorSmoother = scipy.signal.medfilt(Divisor, N_moving_median)

    # Building Harmonic_db and Harmonic_freq
    for n in range(nframes):
        # We want to be able to compute interpolation, so at least size>3, and we want an odd number
        Bw = max(8, 2*int(0.9 * (1/2) * fundamentalList[n] / indexToFreq))
        # We always want Bw>4, and Bw=Bw/2 if k_th is close to Nyquist.

        if missingFundSearch:
            div = DivisorSmoother[n]
            fo = fundamentalList[n] / div
        else:
            fo = fundamentalList[n]
            div = 1

        for h in range(N_h):

            # Theoretical harmonic frequency
            k_th = math.floor((h + div) * fo)
            k_inf = max(0, k_th - int(Bw / 2))
            k_sup = k_inf + int(Bw)

            # If the k_sup is out of the bandwidth, no block method
            if k_th > np.min([f_high/indexToFreq, Nyq]):
                harmonic_db[n, h] = np.min(xdB)
                if n>0:
                    harmonic_freq[n, h] = harmonic_freq[n-1, h]
                else :
                    harmonic_freq[n, h] = fundThroughFrame[n]

            else:
                # Draw the research block
                Block = xdB[k_inf:min(k_sup, Nyq-1), n] # because of the if condition above, block_size > Bw/2 > 4

                # Find the block's maximum
                maxB = max(Block)
                k_maxB = np.argmax(Block, axis=0)

                # Interpolation for the exact maximum
                if 0 < k_maxB <Bw - 1 and ParabolicInterpolation:
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
                # If a silence is detected, the harmonic is kept constant.
                # However, harmonic h must stay higher than the harmonic h-1, so if it's not the case, we raise it to its right place.
                if int_mag < silence_thres and n > 0 and harmonic_freq[n, h] > harmonic_freq[n, max(0,h-1)] and h>0 :
                    harmonic_freq[n, h] = harmonic_freq[n - 1, h]
                    harmonic_db[n, h] = np.min(X_db)
                else:
                    harmonic_freq[n, h] = indexToFreq * int_loc

    return harmonic_freq, harmonic_db


# Smoothing the harmonics trajectories
def smootherHarmonics(harmonic_freq, NmovingMedian):
    harmonic_freqSmoother = harmonic_freq
    for harmo in range(N_h):
        harmonic_freqSmoother[:, harmo] = scipy.signal.medfilt(harmonic_freq[:, harmo], NmovingMedian)
    return harmonic_freqSmoother


def plot_harmonics(harm_freq, harm_freq_smooth):

    plt.figure(figsize=(15, 8))

    plt.subplot(211)
    plt.title('Fundamental and its harmonics - block research method')
    plt.plot(np.arange(len(harm_freq)), harm_freq, '.')
    plt.ylabel("Frequency [Hz]")
    plt.xlabel("Frames")

    plt.subplot(212)
    plt.title('Smoothed fundamental and its harmonics - block research method')
    plt.plot(np.arange(len(harm_freq_smooth)), harm_freq_smooth, '.')
    plt.xlabel("Frames")
    plt.ylabel("Frequency [Hz]")
    plt.tight_layout()

    plt.show()


# Compute and plot the harmonics
if __name__ == "__main__":
    Harmonic_freq, Harmonic_db = findHarmonics_blockMethod(X_db, fundThroughFrameSmoother, indexToFreq,
                                                           MissingFundSearch)
    Harmonic_freqSmoother = smootherHarmonics(Harmonic_freq, N_moving_median)

    plot_harmonics(Harmonic_freq, Harmonic_freqSmoother)

# ------------------------------------------- TRAJECTORIES ------------------------------------------

'''
Building the trajectories according to the evolution of the frequency of each harmonic through time.
The two consecutive frames are set to belong to the same trajectory if the absolute distance between their
corresponding frequencies is less than a predefined value (around a quarter of a tone).
It is assumed that through all frames, even in the moments of inharmonicity or very low amplitude, the sound is
represented by exactly N_h harmonics ([f0, f1, f2...fN_h-1]).
'''


def build_trajectories(Harm_db, Harm_freq):
    '''
    Marks the starting points of new trajectories, according to the distance between consecutive samples.
    Returns the array of marked points (traj), the array of frequencies according to

    :param Harm_db: 2D array of amplitudes in dB of all harmonics through frames
    :param Harm_freq: 2D array of frequencies in Hz of all harmonics through frames
    :return:
        traj: 2D array of marked starting points of new trajectories for each of the harmonic. For each frame,
              takes values of 1 or 2, where the change of the value represents the start of a new trajectory
        traj_freq: 2D array of frequencies through frames, according to whether they belong to a previous trajectory
              or they belong to a newly formed one
        traj_db: 2D array of amplitudes in dB through frame, according to whether they belong to a previous trajectory
              or to a newly formed one
    '''

    traj_index = np.zeros((Harm_db.shape[0], Harm_db.shape[1]))
    traj_freq = np.zeros((Harm_db.shape[0], 2 * Harm_db.shape[1]))
    traj_db = np.zeros((Harm_db.shape[0], 2 * Harm_db.shape[1]))

    for m in range(Harm_db.shape[0] - 1):

        for i in range(Harm_db.shape[1]):

            if m == 0:
                traj_index[m, i] = 1
                traj_freq[m, i * 2] = Harm_freq[m, i]
                traj_db[m, i * 2] = Harm_db[m, i]
            else:
                freq_distance = abs(Harm_freq[m, i] - Harm_freq[m - 1, i])        # measure freq distance
                freq_dev_offset = (Harm_freq[m - 1, i]) * (pow(2, 1 / 24) - 1)    # the quarter of a tone
                if freq_distance < freq_dev_offset:      # if belongs to the same previous trajectory
                    traj_index[m, i] = traj_index[m - 1, i]
                    if traj_freq[m - 1, i * 2] == 0:
                        traj_freq[m, i * 2 + 1] = Harm_freq[m, i]
                        traj_db[m, i * 2 + 1] = Harm_db[m, i]
                    else:
                        traj_freq[m, i * 2] = Harm_freq[m, i]
                        traj_db[m, i * 2] = Harm_db[m, i]
                else:                                    # if new trajectory should begin
                    if traj_index[m - 1, i] == 1:
                        traj_index[m, i] = 2
                    elif traj_index[m - 1, i] == 2:
                        traj_index[m, i] = 1
                    if traj_freq[m - 1, i * 2] == 0:
                        traj_freq[m, i * 2] = Harm_freq[m, i]
                        traj_db[m, i * 2] = Harm_db[m, i]
                    else:
                        traj_freq[m, i * 2 + 1] = Harm_freq[m, i]
                        traj_db[m, i * 2 + 1] = Harm_db[m, i]

    return traj_index, traj_freq, traj_db


def delete_short_trajectories(traj_index, traj_freq, traj_db, Harm_db, min_traj_duration, min_amp_db):
    '''
    Deletes the trajectories whose length is shorter than the predefined min_traj_duration.
    The deleting is done by putting the amplitude in dB of a corresponding sample in Harm_db to a predefined minimum value.

    :param traj, traj_freq, traj_db: Previously found arrays of previously built trajectories
    :param Harm_db: A 2D array of amplitudes in dB of all harmonics through frames
    :param min_traj_duration: predefined number of frames that correspon to a minimum length of a trajectory
    :param min_amp_db: predefined minimum value to assign to the samples that are to be "deleted"
    :return:
        traj, traj_freq, traj_db: Modified arrays after "deleting" short trajectories
        harm_db_filtered: Modified original array of amplitudes in dB, after "deleting" short trajectories
    '''


    harm_db_filtered = np.copy(Harm_db)

    for i in range(traj_index.shape[1]):

        for m in range(traj_index.shape[0] - 1):

            if m == 0:
                traj_start = m
            elif traj_index[m, i] != traj_index[m - 1, i]:
                traj_start = m
            if traj_index[m, i] != traj_index[m + 1, i]:
                traj_end = m
                if (traj_end - traj_start >= 0) & (traj_end - traj_start < min_traj_duration):  # if under predefined minimum length
                    traj_index[traj_start: traj_end + 1, i] = 0
                    traj_freq[traj_start: traj_end + 1, i * 2: i * 2 + 2] = 0
                    traj_db[traj_start: traj_end + 1, i * 2: i * 2 + 2] = min_amp_db
                    harm_db_filtered[traj_start: traj_end + 1, i] = min_amp_db

    return traj_index, traj_freq, traj_db, harm_db_filtered


def cursorMedian(y, order):
    order = max(int(order * len(y)), 1)
    z = np.copy(y)
    k = 0
    while (k + 1) * order < len(y):     # while the remaining length to average is superior to the order
        z[k * order: (k + 1) * order] = np.median(y[k * order: (k + 1) * order])
        k = k + 1
    z[k * order: len(y)] = np.median(y[k * order: len(y)])

    return z


def smooth_trajectories_freq(traj_index, traj_freq, Harm_freq, min_traj_duration):

    Harm_freq_filtered = np.copy(Harm_freq)

    for i in range(traj_index.shape[1]):

        first_start_found = 0

        for m in range(traj_index.shape[0] - 1):

            if (m == 0) & (traj_index[m, i] != 0):
                traj_start = m
                first_start_found = 1
            elif (traj_index[m, i] != 0) & (traj_index[m, i] != traj_index[m - 1, i]):
                traj_start = m
                first_start_found = 1
            if first_start_found & (traj_index[m, i] != traj_index[m + 1, i]):
                traj_end = m
                if (traj_end - traj_start >= 0) & (traj_end - traj_start > min_traj_duration):

                    Harm_freq_filtered[traj_start: traj_end + 1, i] = cursorMedian(Harm_freq[traj_start: traj_end + 1, i], 1)

                    if traj_index[traj_end, i] == 1:
                        traj_freq[traj_start: traj_end + 1, 2 * i] = \
                            cursorMedian(traj_freq[traj_start: traj_end + 1, 2 * i], 1)

                    elif traj_index[traj_end, i] == 2:
                        traj_freq[traj_start: traj_end + 1, 2 * i + 1] =\
                            cursorMedian(traj_freq[traj_start: traj_end + 1, 2 * i + 1], 1)

    return traj_index, traj_freq, Harm_freq_filtered


def plot_trajectories(traj_freq):

    if traj_freq.shape[1] > 0:

        y = np.copy(traj_freq)
        y[y <= 0] = np.nan
        plt.plot(np.arange(len(y)), y, 'k')
        plt.xlabel('Frames')
        plt.ylabel('Frequency[Hz]')
        plt.title('Trajectories')

    plt.tight_layout()
    plt.show()



def subplot_harmonics_intensity(plot, ax, harm_freq, harm_db, n_h, miny, maxy, title):

    x = np.arange(len(harm_freq[:, 0]))
    y = np.copy(harm_freq)
    y = y.reshape(y.size)

    dydx = harm_db
    norm = plot.Normalize(np.min(dydx), np.max(dydx))

    for h in range(n_h):
        y = np.copy(harm_freq[:, h])
        y[np.isnan(y)] = 0
        y[y<=0] = np.nan
        dydx = harm_db[:, h]

        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        lc = LineCollection(segments, cmap='Blues', norm=norm)
        lc.set_array(dydx)
        lc.set_linewidth(2)
        line = ax.add_collection(lc)
        # fig.colorbar(line, ax=ax)

    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(miny, maxy)
    ax.set_title(title)
    ax.set_xlabel("Frames")
    ax.set_ylabel("Frequency [Hz]")

    return ax


def plot_harmonics_intensity(harm_freq, harm_db, traj_freq, traj_db, title2, miny, maxy):

    n_h = harm_db.shape[1]

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    title1 = "Fundamental and its harmonics"
    ax1 = subplot_harmonics_intensity(plt, ax1, harm_freq, harm_db, n_h, miny, maxy, title1)
    ax2 = subplot_harmonics_intensity(plt, ax2, traj_freq, traj_db, n_h*2, miny, maxy, title2)
    plt.show()

if __name__ == "__main__":

    minAmp_db = np.min(Harmonic_db)

    # Computing trajectories
    trajectories, trajectories_freq, trajectories_db = build_trajectories(Harmonic_db, Harmonic_freqSmoother)

    # Plot the trajectories, without representing intensity
    plot_trajectories(trajectories_freq)

    # Deleting short trajectories (below specific minimum duration)
    trajectories, trajectories_freq, trajectories_db, Harmonic_db_filtered =\
        delete_short_trajectories(trajectories, trajectories_freq, trajectories_db, Harmonic_db, minTrajDuration
                                  , minAmp_db)

    # Plot the trajectories, with intensity

    min_y, max_y = np.min(Harmonic_freq), np.max(Harmonic_freq)
    title_traj = "Fundamental and its harmonics after deleting short trajectories"

    plot_harmonics_intensity(Harmonic_freqSmoother, Harmonic_db, trajectories_freq, trajectories_db, title_traj, min_y, max_y)

    # Smoothening out frequency variation within each trajectory
    trajectories, trajectories_freq, Harmonic_freqMedian =\
        smooth_trajectories_freq(trajectories, trajectories_freq, Harmonic_freqSmoother, minTrajDuration)
    # Harmonic_freqSmoother should be used for resynthesis
    # Harmonic_freq_Median should be used for fundamental_synthesis

# ------------------------------------------ PART 2 : SYNTHESIS ------------------------------------------


def linear_interpolation(a, b, n):

    '''
    Linearly interpolating n samples from value a to value b
    :param a: starting value
    :param b: ending value
    :param n: number of samples to put between the starting and the ending value
    '''
    if n-1:
        s = np.arange(n) * (b - a) / (n - 1) + np.ones(n) * a
    else:
        s = a
    return s


def oscillators_bank_synthesis(harm_amp, harm_freq, f_s, hop_length):

    num_frames = harm_amp.shape[0]
    n_h = harm_amp.shape[1]

    # Time axis
    time = np.arange(0, hop_length * num_frames)

    # Allocate the output vector
    out_bankosc = np.zeros(num_frames * hop_length)

    for i in range(n_h):

        # Generate the interpolated amp and freq, for samples within each frame

        IntAmp = np.zeros(num_frames * hop_length)
        IntFreq = np.zeros(num_frames * hop_length)
        IntPhase = np.zeros(num_frames * hop_length)

        for m in range(num_frames - 1):
            IntAmp[m * hop_length:(m + 1) * hop_length] = linear_interpolation(harm_amp[m, i], harm_amp[m + 1, i],
                                                                               hop_length)

            IntFreq[m * hop_length:(m + 1) * hop_length] = np.ones(hop_length) * harm_freq[m, i]

            if m == 0:
                IntPhase[m * hop_length:(m + 1) * hop_length] = np.zeros(hop_length)
            else:
                IntPhase[m * hop_length:(m + 1) * hop_length] = IntPhase[(m-1) * hop_length:m * hop_length] + \
                                (np.ones(hop_length) * 2 * np.pi * harm_freq[m-1, i] * (hop_length+1) * m / f_s) - \
                                (np.ones(hop_length) * 2 * np.pi * harm_freq[m, i] * (hop_length+1) * m / f_s)

        oscillator = IntAmp * np.sin(2 * np.pi * IntFreq * time / f_s + IntPhase)

        out_bankosc = out_bankosc + oscillator

    # Normalizing out
    out_bankosc = out_bankosc * (np.max(audio) - np.min(audio)) / (np.max(out_bankosc) - np.min(out_bankosc))
    return out_bankosc


def adsr_amp(traj, harm_amp, attack, decay, sustainAmp):
    '''
    Defines the amplitude of each trajectory according to the ADSR parameters attack, decay and sustain

    :param traj: The array of trajectories, after deleting short trajectories and smoothening the longer ones
    :param harm_amp: A 2D array of amplitudes of all harmonics through frames
    :param attack: Attack time in seconds - time until reaching the maximum amplitude in the trajectory
    :param decay: Decay time in seconds - time for reaching the constant sustain amplitude after the maximum amplitude point
    :param sustainAmp: The final amplitude of the trajectory after the decay time
    '''

    harm_amp_adsr = np.copy(harm_amp)
    attack_frames = round(attack * Fs / Hop_length)
    decay_frames = round(decay * Fs / Hop_length)

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
                if (traj_end - traj_start >= 0) & (traj[traj_start, i] != 0):

                    # the amplitude of the highest peak(after the attack time) is set to the max amplitude within the trajectory
                    max_value_amp = np.max(harm_amp[traj_start: traj_end + 1, i])

                    # if the trajectory is longer than the attack time
                    if traj_start + attack_frames < traj_end:
                        harm_amp_adsr[traj_start: traj_start + attack_frames + 1, i] =\
                            linear_interpolation(0.1, max_value_amp, attack_frames + 1)

                        # if the trajectory is longer than the sum of attack time and decay time
                        if traj_start + attack_frames + decay_frames < traj_end:
                            harm_amp_adsr[traj_start + attack_frames + 1: traj_start + attack_frames + decay_frames + 1, i] = \
                                linear_interpolation(max_value_amp, sustainAmp, decay_frames)
                            harm_amp_adsr[traj_start + attack_frames + decay_frames + 1: traj_end + 1, i] = \
                                np.ones(traj_end - traj_start - attack_frames - decay_frames) * sustainAmp;

                        else:
                            harm_amp_adsr[traj_start + attack_frames + 1: traj_end + 1, i] =\
                                linear_interpolation(max_value_amp, sustainAmp, traj_end - traj_start - attack_frames)

                    # if the trajectory is shorter than the attack time, interpolate the whole trajectory from
                    # the minimum value to the highest peak
                    else:
                        harm_amp_adsr[traj_start: traj_end + 1, i] = \
                            linear_interpolation(0.1, max_value_amp, traj_end - traj_start + 1)

    return harm_amp_adsr


# ------------------------------------------ SOUND - WAV FILE CREATION ------------------------------------------


def wave_file_creation(out_bankosc, f_s, file_path):
    # Open the wav file
    wav_file = wave.open(file_path, "w")
    print("Saving " + file_path + "...")

    # Writing parameters
    data_size = len(out_bankosc)
    amp = 64000.0  # multiplier for amplitude
    nchannels = 1
    sampwidth = 2  # 2 for stereo
    comptype = "NONE"
    compname = "not compressed"

    # Set writing parameters
    wav_file.setparams((nchannels, sampwidth, f_s, data_size, comptype, compname))

    # Write out in the wav file
    for s in out_bankosc:
        wav_file.writeframes(struct.pack('h', int(s * amp / 2)))

    wav_file.close()
    print(file_path + " saved")


def resynthesis(harm_db, harm_freq, fs, hop_length, path_name):
    harm_amp = librosa.db_to_amplitude(harm_db)
    bankosc = oscillators_bank_synthesis(harm_amp, harm_freq, fs, hop_length)

    # Plotting the original and the re-synthesised audio files in time domain
    plot_synthesis(audio, bankosc, "Re-synthesized audio file")

    # Writing the file with controlled harmonics
    wave_file_creation(bankosc, fs, path_name)

    return bankosc


def custom_synthesis(harm_db, harm_freq, harm_orig_db, harm_orig_freq, traj, amplitudes_array, inharm_array, attack,
                               decay, sustain_ampl, fs, hop_length, envelope_adsr, path_name):

    harmonic_freq_additive = np.zeros((harm_freq.shape[0], harm_freq.shape[1]))
    fund_freq = harm_freq[:, 0]

    harmonic_amp = librosa.db_to_amplitude(harm_db)
    harmonic_amp[harmonic_amp <= 0.0001] = 0
    fund_amp = harmonic_amp[:, 0]

    harmonic_amp_additive = np.zeros((harm_db.shape[0], harm_db.shape[1]))

    for i in range(harmonic_amp_additive.shape[1]):
        harmonic_amp_additive[:, i] = fund_amp * amplitudes_array[i]
        harmonic_freq_additive[:, i] = fund_freq * (i + 1) * (1 + inharm_array[i] * (pow(2, 1/24) - 1))

    if envelope_adsr:
        harmonic_amp_additive = adsr_amp(traj, harmonic_amp_additive, attack, decay, sustain_ampl)

    traj_add, traj_add_freq, traj_add_db = build_trajectories(librosa.amplitude_to_db(harmonic_amp_additive),
                                                              harmonic_freq_additive)

    min_y, max_y = np.min(harmonic_freq_additive), np.max(harmonic_freq_additive)

    title_traj_median = "Fundamental and harmonics after the custom synthesis"
    plot_harmonics_intensity(harm_orig_freq, harm_orig_db, traj_add_freq, traj_add_db,
                             title_traj_median, min_y, max_y)

    bankosc = oscillators_bank_synthesis(harmonic_amp_additive, harmonic_freq_additive, fs, hop_length)

    # Plotting the original and customarily synthesised audio files in time domain
    plot_synthesis(audio, bankosc, "Customarily synthesized audio file")

    # Writing the file with controlled harmonics
    wave_file_creation(bankosc, fs, path_name)

    return bankosc


def plot_synthesis(audio_orig, audio_synth, title):

    plt.figure(figsize=(10, 4))
    plt.subplot(211)
    plt.title('Original audio file')
    plt.plot(np.arange(len(audio_orig)), audio_orig)
    plt.ylabel("Amplitude")
    plt.xlabel("Time [s]")

    plt.subplot(212)
    plt.title(title)
    plt.plot(np.arange(len(audio_synth)), audio_synth)
    plt.ylabel("Amplitude")
    plt.xlabel("Time [s]")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":  # For now the whole synthesis part doesn't exists outside of main

    # Calling both ways of synthesis
    # 1. Resynthesis: Part with tracking the original harmonics
    # 2. Synthesis from fundamental: Part with controlling harmonics

    file_path1 = "..\\synthesized_sound\\Synthesized_" + "example_" + str(example_number) + ".wav"
    bankosc_resynth = resynthesis(Harmonic_db_filtered, Harmonic_freqSmoother, Fs, Hop_length, file_path1)

    file_path2 = "..\\synthesized_sound\\Synthesized_" + "custom_example_" + str(example_number) + ".wav"
    bankosc_custom_synth = custom_synthesis(Harmonic_db_filtered, Harmonic_freqMedian, Harmonic_db,
                                            Harmonic_freqSmoother, trajectories, Amplitudes_array,
                                            Inharmonicity_array, attack_sec, decay_sec, sustain_amp, Fs, Hop_length,
                                            envelopeADSR, file_path2)
