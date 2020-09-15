# TonalComp

#### An application for the analysis and synthesis of monophonic sounds

## Introduction
This didactic application was built in the frame of the course Sound Analysis, Synthesis and Processing. 
It performs the analysis and re-synthesis of any monophonic sound, and aims to outline to the user the different steps and parameters of this process.
The analysis is based on peak tracking, while the synthesis is performed with oscillators, with a possibility to customize the final sound.

## User interface
The first window of the welcome page shows the different examples of monophonic sounds that can be selected.
The second window gather the analysis parameters. The user must indicate its estimation of the frequency bandwidth of the chosen sound, so that the windowing parameters can be computed. The user can overwrite some of those parameters, such as the window's and the stft's length.

Once the analysis has been launched, several plots show the user the computed fundamental, harmonics and trajectories.


## Analysis
#### Windowing and STFT
The audio file is analysed with a hamming window, parameterized thanks to the user input.  
The window's length is chosen as small as possible to have the best temporal resolution, while respecting a frequency resolution inferior to f_low.  
The number of samples of the STFT is chosen to respect the Just Noticeable Difference criteria.  
The Short Time Fourier Transform is then computed.

#### Fundamental
In oder to find the fundamental, a peak tracking is conducted on each frame of the stft by the function scipy.signal.find_peaks. It is parametered to find peaks separated by a gap slightly inferior to f_low.
The peak of smallest frequency above f_low is identified as the fundamental of the frame.  

The fundamental is then smoothed with a median filter to discard anomalies.

#### Harmonics
Thanks to the fundamental value, the theoretical frequency of each harmonic is computed. The exact frequency is obtained by looking for the maximum magnitude of the stft around this theoretical frequency, and precising this maximum with a parabolic interpolation.
When silences are detected, the research of a maximum is deactivated and frequencies are kept constant.  

The harmonics are also smoothed with a median filter to discard anomalies.  

If the user has activated the research of a missing fundamental, such as for a 300 Hz sound high-pass filtered at 400 Hz, it means the fundamental found previously was actually a harmonic. 
Therefore, an algorithm looks for the real value of the fundamental, to compute the right theoretical frequencies of the harmonics.


#### Trajectories
Each harmonic line is deducted into several trajectories, with an end and a beginning in time. The shortest trajectories are interpreted as anomalies, and are discarded. 


## Synthesis
The synthesis is computed by adding oscillators, which frequency and amplitude are driven by the trajectories frequencies and amplitude.
A first re-synthesis aims to reproduce exactly the original sound. 

A second synthesis is performed only considering the fundamental. The timbre is customized by the user, who can choose the harmonics' relative amplitudes and the adsr envelope. 

## Files

- **main.py**   
Main script of the application.

- **audioAnalysis.py**   
Module that contains the class definitions.

- **gui.py**  
Module that builds the graphical user interface.

- **demo_sound**  
Bank of monophonic sound examples in the wav format

## Authors
Clément Jameau  
Aliette Ravillion  
Andriana Takic  