import numpy as np
import librosa.display
import math
import matplotlib.pyplot as plt
import glob
import librosa.display

demo_files = []
for file in glob.glob("..\\demo_sound\\*.wav"):
    demo_files.append(file)

print(demo_files)

# example_number = int(input(".wav example number = "));
example_number = 9
path_name = demo_files[example_number - 1]
audio, Fs = librosa.load(path_name, sr=None)
print("Opening " + path_name)
print("Fs: ", Fs)

# ------------------------------------------ WINDOWING ------------------------------------------
# Bandwidth given by the user
# f_low = int(input("lowest frequency (above 30Hz) = "));
# f_high = int(input("highest frequency = ")); #bandwidth = [f_low, f_high]
f_low = 10  # will limit d_f, shouldn't be put under 30Hz in the app

# Number of harmonics
N_h = 10

# Window type
Win_type = "hamming"
# Smoothness factor
L = 4

# Frequency separation
d_fmin = 30  # DO NOT CHANGE - The frequency resolution is limited, to balance with the temporal resolution
d_f = f_low
if f_low < d_fmin:
    d_f = d_fmin
    print('\033[93m' + f"WARNING: f_low chosen too low, and therefore changed to {d_fmin}" + '\033[0m')

# Window length
# Depends on our sampling frequency and the desired frequency separation
Win_length = math.floor(L * Fs / d_f)

# Number of FFT samples
# N_fft should be at least the window's length, and should respect the JND criteria
N_fft = max(2 ** math.ceil(math.log2(Win_length)), 2 ** math.ceil(math.log2(Fs / (2 * 3))))

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

print("Peak resolution d_f = " + str(d_f) + " Hz")
print("N_fft = " + str(N_fft))
print("Window length = " + str(Win_length))
print("Hop length = " + str(Hop_length))
print("n frames = " + str(n_frames))
print("Audio length = " + str(len(audio)))


# ------------------------------------------ STFT ------------------------------------------

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
    librosa.display.specshow(X_db, y_axis='log', x_axis='time', sr=Fs)

    plt.colorbar(format='%+2.0f dB')
    plt.title('Frequency spectrogram')
    plt.tight_layout()
    plt.show()


# ------------------------------------------ PEAK TRACKING ------------------------------------------

# peakFinding goes through all frames of xdB. For each frame, it keeps the loudest frequency's localization and magnitude.
# If the peak's magnitude is too low (<-25 or -35), it is interpreted as a silence. The pitch is considered as the same as the precedent frame, but silent.
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


peakLoc_List = []  # Too constraining to initialise with numpy
peakMag_List = []
X_db_actualStep = X_db.copy()

numberOfPeaks = 5

# for each trajectory : find the maximum trajectory, save it and erase it in xdb.
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


# ------------------------------------------ FUNDAMENTAL TRAJECTORY PRINTING ------------------------------------------

N_moving_median = 5
fundThroughFrame = np.amin(peakLoc_List, axis=0)
fundThroughFrameSmoother = movingMedian(fundThroughFrame, windowLength=N_moving_median)
indexToFreq = Fs / N_fft

if __name__ == "__main__":
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
    value = gamma
    if alpha - 2 * beta + gamma != 0:
        location = 0.5 * (alpha - gamma) / (alpha - 2 * beta + gamma)
        value = beta - value * (alpha - gamma) / 4
    return [value, location]

# Check if peaks are spaced by the computed fundamental, or by one of its divisors
def real_fundamental(peak_freqs, fo):
    peak_freqs2=peak_freqs.copy()
    peak_freqs2.sort()
    gap = np.min([np.abs(peak_freqs2[0:len(peak_freqs2)-1]-peak_freqs2[1:len(peak_freqs2)])]) # we expect gap = fo if fo is the right fundamental
    # A few cases : 1) Gap ~ fo, then fo=fo/1   2) Gap ~ fo/n, then fo=fo/n, 3) Gap = 2fo, then min([fo, 1.5fo,..])
    divisor = 1
    while np.abs(fo/divisor-gap)>np.abs(fo/(divisor+1)-gap) and divisor<10:
        divisor = divisor+1
    fo = fo/divisor
    return fo

# Width of the research block
Bw = 2 * MainLobe

# Initialization of storage vectors
Harmonic_db = np.zeros((n_frames, N_h))
Harmonic_freq = np.zeros((n_frames, N_h))

for n in range(n_frames):

    fo = real_fundamental(peakLoc_List[:,n], fundThroughFrame[n])

    for h in range(2, N_h + 2):

        # Theoretical harmonic frequency
        k_th = math.floor(h * fo)

        # If the theoretical harmonic frequency is under 20 000Hz, we can apply the block method
        if k_th * indexToFreq > 20000:
            Harmonic_db[n, h - 2] = 0
            if n>1:
                Harmonic_freq[n, h - 2] = Harmonic_freq[n - 1, h - 2]
            else:
                Harmonic_freq[n, h - 2] = 0

        else :
            # Draw the research block
            k_inf = max(0, k_th - Bw)
            k_inf = min(k_inf, Nyq)
            Block = X_db[k_inf:min(k_inf + 2 * Bw - 1, Nyq), n]

            maxB = max(Block)
            k_maxB = np.argmax(Block, axis=0)

            # Interpolation
            if 0 < k_maxB < 2 * (Bw - 1):
                alpha = Block[k_maxB - 1]
                beta = Block[k_maxB]
                gamma = Block[k_maxB + 1]
                [int_mag, int_loc] = parabolic_interpolation(alpha, beta, gamma)
            else:
                [int_mag, int_loc] = [maxB, k_maxB]
            int_loc = k_inf + k_maxB + int_loc

            # Store the interpolated peak
            Harmonic_db[n, h - 2] = int_mag
            if int_mag < -35 and n > 0:  # same idea than above : -35 are interpreted as silence, the pitch remains the same than before
                Harmonic_freq[n, h - 2] = Harmonic_freq[n - 1, h - 2]
            else:
                Harmonic_freq[n, h - 2] = indexToFreq * int_loc

# Smoothing the harmonics trajectories
Harmonic_freqSmoother = Harmonic_freq.copy()
for h in range(2, N_h + 2):
    Harmonic_freqSmoother[:, h - 2] = movingMedian(Harmonic_freq[:, h - 2], windowLength=N_moving_median)

# Plot the harmonics trajectories
if __name__ == "__main__":
    plt.figure(figsize=(15, 8))

    plt.subplot(211)
    plt.title('Fundamental and its harmonics - block research method')
    plt.plot(np.arange(len(fundThroughFrame)), indexToFreq * fundThroughFrame, color = "black")
    plt.plot(np.arange(len(Harmonic_freq)), Harmonic_freq)
    plt.ylabel("Hz")
    plt.xlabel("Time")

    plt.subplot(212)
    plt.title('Smoothed fundamental and its harmonics - block research method')
    plt.plot(np.arange(len(fundThroughFrameSmoother)), indexToFreq * fundThroughFrameSmoother, color = "black")
    plt.plot(np.arange(len(Harmonic_freqSmoother)), Harmonic_freqSmoother)
    plt.xlabel("Time")
    plt.ylabel("Hz")
    plt.tight_layout()

    plt.show()

# ------------------------------------------ PART 2 : SYNTEHSIS ------------------------------------------

# Synthesis parameters

# Synthesis window = analysis window
WinSynth=np.hamming(Win_length)

def linear_interpolation(a,b,n):
    s = np.arange(n)*(b-a)/(n-1) + np.ones(n)*a
    return s

# Smallest frequency separation between two notes in the melody
delta_notes = 0.95*abs(d_fmin*2**(1/12)-d_fmin) #taken as (lowest_note + 1/2 tone) - (lowest_note)

# Time axis
win_time=np.arange(0,Win_length)

# Allocate the output vector
out=np.zeros(len(audio))

# Compute the normalized frequency [0,pi]
Harmonic_freqSmootherNorm = 2*np.pi*Harmonic_freqSmoother/Fs

# Additive synthesis
for m in range(n_frames-1):
    buffer = np.zeros(Win_length)

    # Fundamental synthesis
    
    # Harmonics synthesis
    for i in range(N_h):
        # Interpolation of sines amplitude
        win_amp = linear_interpolation(Harmonic_db[m,i], Harmonic_db[m+1,i], Win_length)
        if abs(Harmonic_freqSmoother[m,i]-Harmonic_freqSmoother[m+1,i])<delta_notes :
            win_freq=linear_interpolation(Harmonic_freqSmoother[m,i],Harmonic_freqSmoother[m+1,i],Win_length)
        else :
            win_freq=np.ones(Win_length)*Harmonic_freqSmoother[m,i]
        # Generate sine
        win_sine = win_amp*np.sin(2*np.pi*win_time*win_freq/Fs)
        buffer = buffer+win_sine

    ola_indices_a = (m)*Hop_length
    ola_indices_b = (m)*Hop_length + Win_length
    y=WinSynth*buffer
    out[ola_indices_a:ola_indices_b] = out[ola_indices_a:ola_indices_b]+y[0:len(out[ola_indices_a:ola_indices_b])]

#out=out*normalize(audio)/normalize(out) #NOT WORKING YET

