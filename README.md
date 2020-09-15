# TonalComp

#### An application for the analysis and synthesis of monophonic sounds

## Introduction
This didactic application was built in the frame of the course Sound Analysis, Synthesis and Processing. 
It performs the analysis and re-synthesis of any monophonic sound, and aims to outline to the user the different steps and parameters of this process.
The analysis is based on peak tracking, while the synthesis is performed with oscillators, with a possibility to customize the final sound.

## User interface
The welcome page of user interface has two windows.  
The first one shows the different examples of monophonic sounds that can be selected.  
The second window allows the user to vary the parameters of the analysis.  


## Analysis
### User input
The user must select a sound to analyze, and indicate its estimation of the frequency bandwidth of the chosen sound for the windowing parameters. 
Then, the user can change various windowing and synthesis parameters.

### Windowing
The audio file is analysed with a hamming window, parameterized thanks to the user input.  
The window's length is chosen as small as possible to have the best temporal resolution, while respecting a frequency resolution inferior to f_low.  
The number of samples of the STFT is chosen to respect the Just Noticeable Difference criteria.  
The STFT is computed on the windowed frames.

### Peak tracking
In oder to find the fundamental, a peak tracking is conducted on each frame of the stft by the function scipy.signal.find_peaks. It is parametered to find peaks separated by a gap slightly inferior to f_low.
The peak of smallest frequency above f_low is identified as the fundamental.


## Synthesis


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