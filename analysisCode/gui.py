# The code for changing pages was derived from: http://stackoverflow.com/questions/7546050/switch-between-two-frames-in-tkinter
# Code addapted from tutorial here : https://pythonprogramming.net/tkinter-depth-tutorial-making-actual-program/
# License: http://creativecommons.org/licenses/by-sa/3.0/
import winsound
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib import style

import tkinter as tk
from tkinter import ttk
from threading import Thread

import glob
import time

from matplotlib import pyplot as plt
import numpy as np

import librosa.display
from pathlib import Path
import audioAnalysis


LARGE_FONT = ("Verdana", 12)
NORM_FONT = ("Verdana", 10)
SMALL_FONT = ("Verdana", 8)

# matplotlib.use("TkAgg")
style.use("ggplot")

f = Figure()
librosa_display_subplt = f.add_subplot(111)
# librosa.display.specshow(main.X_db, y_axis='log', x_axis='time', ax=librosa_display_subplt)
myaudio = type('AudioAnalysis', (), {})()
myaudio = None

tutorial_parameters_text = \
    "The first step of the analysis is to process the audio input through a Short Time Fourier Transform. \n" \
    "Some parameters of the windowing and of the FFT highly influence the accuracy of the analysis. For example: \n\n" \
    " Window \n" \
    "- The shortest the analysis window is, the better the time resolution of the analysis. \n" \
    "- The longest the analysis window is, the smaller gets the minimum difference there has to be between \n" \
    "  2 peaks to distinguish them. Since we are working on monophonic harmonic sounds, two consecutive peaks are two consecutive harmonics of the sound. \n\n" \
    "FFT length \n" \
    "- The length of this windows is zero padded to a power of two to get the length of the FFT,\n" \
    "  which defines the frequency resolution of the spectrum of this each window of the signal.\n" \
    "- If the length of the fft doesn't meet the Just Noticeable Difference criteria, it will be \n" \
    "   automatically increased until the frequency resolution is lower than the JND. \n\n" \
    "- Be aware of possible memory errors in case of too long fft. \n\n" \
    "Bandwidth \n" \
    "- f_min and f_high delimit a search area for the fundamental and harmonics' frequencies. \n" \
    "- f_min also determines the peak resolution, since the window is designed to resolve peaks spaced by f_min. \n" \
    "- f_min is finally used in the peak finding function, that can't find peaks (harmonics) spaced by a distance shorter than f_min. \n" \



# simple function that popup a message in a dialog window.
def popupmsg(msg):
    popup = tk.Tk()
    popup.wm_title("!")
    label = ttk.Label(popup, text=msg, font=NORM_FONT)
    label.pack(side="top", fill="x", pady=10)
    b1 = ttk.Button(popup, text="Okay", command=popup.destroy)
    b1.pack()
    popup.mainloop()


# function that prepare the functionnality "show audio frame" by checking the validity of the parameter num_value
# which is the frame number that we want to see
def show_audio_frame(tk_win, num_value):
    tk_win.destroy()
    try:
        value = int(num_value)
    except ValueError:
        popupmsg("Invalid number format")
        return

    if 0 <= value < myaudio.n_frames:
        myaudio.showframe(value)
    else:
        popupmsg("Selected number out of range")


def ask_frame_num():
    if myaudio is None:
        popupmsg("ERROR : no audio file processed currently")
    else:
        num = tk.Tk()
        num.wm_title("Window selection")
        label = ttk.Label(num, text=f"select your window's number from 0 to {myaudio.n_frames-1}", font=NORM_FONT)
        label.pack(side="top", fill="x", pady=10)

        num_value = ttk.Entry(num)
        num_value.pack()

        b1 = ttk.Button(num, text="Ok", command=lambda: show_audio_frame(num, num_value.get()))
        b1.pack()
        num.mainloop()


def show_fundamental():
    if myaudio is None:
        popupmsg("ERROR : no audio file processed currently")
    else:
        myaudio.showfundamental()


def show_trajectory():
    if myaudio is None:
        popupmsg("ERROR : no audio file processed currently")
    else:
        myaudio.show_trajectories()


def prepare_resynthesis():
    if myaudio is None:
        popupmsg("ERROR : no audio file processed currently")
    else:
        myaudio.resynthesize()


# function that resets the instance myaudio (class AudioAnalysis) before going back to the start page.
def backtolobby(parent):
    global myaudio
    myaudio = None
    parent.show_frame(StartPage)


# Implementation of the application, inheritting from tkinter class Tk
class TonalCompGui(tk.Tk):

    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)

        # tk.Tk.iconbitmap(self, default="clienticon.ico")
        tk.Tk.wm_title(self, "TonalComp")

        container = tk.Frame(self)
        container.pack(side="top", fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        # Menu bar
        menubar = tk.Menu(container)
        filemenu = tk.Menu(menubar, tearoff=0)
        filemenu.add_command(label="Back to lobby", command=lambda: backtolobby(self))
        filemenu.add_separator()
        filemenu.add_command(label="Exit", command=self.destroy)    # "command = quit" causes issues
        menubar.add_cascade(label="File", menu=filemenu)

        analysismenu = tk.Menu(menubar, tearoff=0)
        analysismenu.add_command(label="show frame", command=ask_frame_num)
        analysismenu.add_command(label="show fundamental", command=show_fundamental)
        analysismenu.add_command(label="show final trajectories", command=show_trajectory)
        menubar.add_cascade(label="Analysis", menu=analysismenu)

        synthmenu = tk.Menu(menubar, tearoff=0)
        synthmenu.add_command(label="re-synthesis", command=prepare_resynthesis)
        synthmenu.add_command(label="custom synthesis", command=lambda: self.show_frame(CustomSynthesisPage))
        menubar.add_cascade(label="Synthesis", menu=synthmenu)

        tk.Tk.config(self, menu=menubar)

        # Creating the pages that will compose the application, and adding it to the dictionnary self.frames
        self.frames = {}

        for F in (StartPage, LoadingPage, MainPage, CustomSynthesisPage):
            frame = F(container, self)

            self.frames[F] = frame

            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame(StartPage)

    # Method of our app that permits to navigate from one frame to another
    def show_frame(self, cont):
        if cont == MainPage:    # This is the wrong place to do it but I didn't find better yet
            self.frames[cont].infosText.set(myaudio.pathName+"\n\nFs: "+str(myaudio.Fs)+"\nN_fft: "+str(myaudio.N_fft) +
                                            "\nN_frames: "+str(myaudio.n_frames))
        frame = self.frames[cont]
        frame.tkraise()


# "Lobby", or starting page that shows up when executing the app. In here you will select your audio file to analyse
# and chose some parameters of the analysis.
class StartPage(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.grid_columnconfigure(0, weight=1)
        label = tk.Label(self, text=("Hi and welcome to TonalComp. \nSelect an audio file, "
                                     "or pick an example"), font=LARGE_FONT)
        label.grid(row=0, pady=(10, 30), sticky="nsew")

        listbox = tk.Listbox(self, height=10)
        listbox.grid(row=1, padx=50, sticky="nsew")

        self.selectedPath = ""

        for file in glob.glob("..\\demo_sound\\*.wav"):
            listbox.insert(-1, file)

        def selectItem(evt):
            self.selectedPath = listbox.get(listbox.curselection())
            self.labelText.set("Audio file : " + self.selectedPath)
            self.button3['state'] = 'normal'
        listbox.bind("<<ListboxSelect>>", selectItem)

        button1 = ttk.Button(self, text="Select your own file file", command=lambda: self.selectFile())
        button1.grid(row=2, padx=50, sticky="e")

        self.labelText = tk.StringVar()
        self.labelText.set("no audio file selected")
        label2 = tk.Label(self, textvariable=self.labelText, font=NORM_FONT)
        label2.grid(row=3, padx=20, sticky="nsew")

        self.paramBg = "grey70"
        parametersFrame = tk.Frame(self, height=150, background=self.paramBg)
        parametersFrame.grid(row=4, padx=100, pady=20)

        analParamLabel = tk.Label(parametersFrame, text="Analysis parameters", font=LARGE_FONT, background=self.paramBg)
        analParamLabel.grid(row=0, sticky="nsew", padx=15, pady=15)

        tk.Label(parametersFrame, text="window's length multiplicator (>1)", background=self.paramBg).grid(row=1, sticky="e")
        tk.Label(parametersFrame, text="fft length multiplicator", background=self.paramBg).grid(row=2, sticky="e")

        self.winLength_mul_str = tk.StringVar()
        self.nfft_mul_str = tk.StringVar()
        self.fmin_str = tk.StringVar()
        self.fmax_str = tk.StringVar()
        self.minTrajSec = tk.StringVar()

        self.winLength_mul_str.set("1")
        # choices = ['0.25', '0.5', '1', '2', '4']
        choices = ['1']
        self.nfft_mul_str.set('1')
        self.fmin_str.set("150")
        self.fmax_str.set("18000")
        self.minTrajSec.set("0.1")
        e1 = tk.Entry(parametersFrame, textvariable=self.winLength_mul_str, width=15)
        e2 = tk.OptionMenu(parametersFrame, self.nfft_mul_str, *choices)
        e1.grid(row=1, column=1, sticky="w", padx=5)
        e2.grid(row=2, column=1, sticky="w", padx=5)

        tk.Label(parametersFrame, text="Bandwidth", font=NORM_FONT, background=self.paramBg).grid(row=3, sticky="e", pady=(10, 0))

        tk.Label(parametersFrame, text="f_min", background=self.paramBg).grid(row=4, sticky="e")
        tk.Label(parametersFrame, text="f_max (no space)", background=self.paramBg).grid(row=5, sticky="e", pady=(0, 20))

        e3 = tk.Entry(parametersFrame, textvariable=self.fmin_str, width=15)
        e4 = tk.Entry(parametersFrame, textvariable=self.fmax_str, width=15)
        e3.grid(row=4, column=1, sticky="w", padx=5)
        e4.grid(row=5, column=1, sticky="w", padx=5, pady=(0, 20))
        tk.Label(parametersFrame, text="Hz", background=self.paramBg).grid(row=4, column=2, sticky="w", padx=(0, 20))
        tk.Label(parametersFrame, text="Hz", background=self.paramBg).grid(row=5, column=2, sticky="w", padx=(0, 20), pady=(0, 20))

        tk.Label(parametersFrame, text="Smoothering", font=NORM_FONT, background=self.paramBg).grid(row=6, sticky="e", pady=(10, 0))

        tk.Label(parametersFrame, text="minimum trajectory length", background=self.paramBg).grid(row=7, sticky="e", pady=(0, 20))

        e5 = tk.Entry(parametersFrame, textvariable=self.minTrajSec, width=15)
        e5.grid(row=7, column=1, sticky="w", padx=5, pady=(0, 20))
        sec = tk.Label(parametersFrame, text="seconds", background=self.paramBg)
        sec.grid(row=7, column=2, sticky="w", padx=(0, 20), pady=(0, 20))

        noFundBool = tk.BooleanVar()
        noFundCheckBx = tk.Checkbutton(parametersFrame, text="Consider the possibility of a missing fundamental "
                                                             "frequency", variable=noFundBool, background=self.paramBg)
        noFundCheckBx.grid(row=8, column=0, columnspan=3, sticky="w", padx=(15, 30), pady=5)

        buttonMore = ttk.Button(parametersFrame, text="Learn more about these parameters",
                                command=lambda: popupmsg(tutorial_parameters_text))
        buttonMore.grid(row=9, column=0, columnspan=3)

        def goAction():
            controller.show_frame(LoadingPage)
            controller.update()
            global myaudio
            # app.config(cursor="wait") # not working
            time_1 = time.time()
            try:
                print(float(self.winLength_mul_str.get()))
                print(float(self.nfft_mul_str.get()))
                analysisParams = audioAnalysis.AnalysisParameters(int(self.fmin_str.get()), int(self.fmax_str.get()),
                                                                  float(self.minTrajSec.get()), noFundBool.get())
                myaudio = audioAnalysis.AudioAnalysis(self.selectedPath, analysisParams, float(self.winLength_mul_str.get()),
                                                      float(self.nfft_mul_str.get()))
            except ValueError:
                controller.show_frame(StartPage)
                popupmsg("ERROR : Analysis parameters entry format error or in any way too presumptuous")

            print(f"\n\nComputation in {time.time()-time_1} seconds")
            time_2 = time.time()
            librosa_display_subplt.clear()
            librosa.display.specshow(myaudio.X_db, sr=myaudio.Fs, y_axis='log', x_axis='frames', ax=librosa_display_subplt)
            plt.title('Frequency spectrogram')  # not working
            print(f"\n\nLibrosa display computation in {time.time()-time_2} seconds")
            time_3 = time.time()
            f.canvas.draw()
            f.canvas.flush_events()
            # app.config(cursor="")     # not working
            controller.show_frame(MainPage)
            print(f"\n\nCanvas drawing in {time.time()-time_3} seconds")

        self.button3 = ttk.Button(self, text="GO", width=50, state=tk.DISABLED, command=goAction)
        self.button3.grid(row=5, padx=50, pady=10)

    def selectFile(self):
        filename = tk.filedialog.askopenfilename(filetypes=(("Audio files (wav,m4a)", "*.wav;*.m4a"),
                                                    ("All files", "*.*")))  # mp3 not supported yet (can be with ffmpeg)
        if filename:
            try:
                if Path(filename).stat().st_size < 10000000:    # if file size < 10MB
                    self.selectedPath = filename
                    self.labelText.set("Audio file : " + self.selectedPath)
                    self.button3['state'] = 'normal'
                else:
                    popupmsg("ERROR : File size > 10MB")
            except ValueError:
                popupmsg("ERROR in the file")
                return


# Loading page that distracts the user during the analysis computation.
class LoadingPage(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="Loading...", font=LARGE_FONT)
        label.pack(pady=10, padx=10)


# Main page that appears ones STFT of the audio file is computed. From here you can navigate through differents options
# of the analysis and the synthesis
class MainPage(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        frame1 = tk.Frame(self)
        frame1.pack(fill=tk.X)
        label = tk.Label(frame1, text="Main Page", font=LARGE_FONT)
        label.pack(pady=10, padx=10, side=tk.TOP)

        def _playSound():
            playButton['state'] = 'disabled'
            winsound.PlaySound(myaudio.pathName,  winsound.SND_FILENAME)
            playButton['state'] = 'normal'

        def threadPlaySound():
            Thr = Thread(target=_playSound)
            Thr.start()  # Launch created thread

        playButton = ttk.Button(frame1, text='Play', command=threadPlaySound)
        playButton.pack(pady=20, padx=50, side=tk.LEFT)

        self.infosText = tk.StringVar()
        self.infosText.set('waiting to load audio....................................................')
        w = tk.Message(frame1, textvariable=self.infosText, justify=tk.RIGHT, width=450)
        w.pack(pady=20, padx=50, side=tk.RIGHT, fill=tk.BOTH)

        frame2 = tk.Frame(self)
        frame2.pack(fill=tk.BOTH, expand=True)
        # We need to set a canva to be able to integrate any matplotlib printing in our frame
        canvas = FigureCanvasTkAgg(f, frame2)
        canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
        canvas.draw()

        toolbar = NavigationToolbar2Tk(canvas, frame2)
        toolbar.update()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)


# Page from which we set all parameters for the custom synthesis (harmonics amplitude, inharmonicity, ADSR enveloppe)
class CustomSynthesisPage(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)

        titleLabel = tk.Label(self, text="CustomSynthesis", font=LARGE_FONT)
        titleLabel.grid(row=0, column=0, columnspan=12, sticky='nsew')

        def backCommand():
            if myaudio is None:
                controller.show_frame(StartPage)
            else:
                controller.show_frame(MainPage)

        backButton = ttk.Button(self, text='Back', command=backCommand)
        backButton.grid(row=1, column=0, padx=10)

        subtitleLabel = tk.Label(self, text="custom your harmonics", font=NORM_FONT)
        subtitleLabel.grid(row=1, column=3, columnspan=6, pady=20)

        labelAmplitude = tk.Label(self, text="Amplitude", font=SMALL_FONT)
        labelAmplitude.grid(row=3, column=0, padx=10, pady=10)
        labelInharmo = tk.Label(self, text="Inharmonicity", font=SMALL_FONT)
        labelInharmo.grid(row=4, column=0, padx=10, pady=10)

        harm_number = 12

        amplitudeVarList = []
        inHarmonicityVarList = []
        for i in range(harm_number):
            amplitudeVarList.append(tk.DoubleVar())
            inHarmonicityVarList.append(tk.DoubleVar())
            amplitudeVarList[i].set(1/(i+1))
            inHarmonicityVarList[i].set(0)

            harmoScale = ttk.Scale(self, orient='vertical', from_=1, to=0, variable=amplitudeVarList[i])
            harmoScale.grid(row=3, column=i+1, padx=30)
            labelValue = tk.Label(self, textvariable=amplitudeVarList[i], font=SMALL_FONT, width=4, anchor='w')
            labelValue.grid(row=3, column=i+1, padx=15)

            inharmonicityScale = ttk.Scale(self, orient='horizontal', from_=-1, to=1, length=50, variable=inHarmonicityVarList[i])
            inharmonicityScale.grid(row=4, column=i+1, padx=10)
            labelValue = tk.Label(self, textvariable=inHarmonicityVarList[i], font=SMALL_FONT, width=4, anchor='w')
            labelValue.grid(row=5, column=i+1, padx=10)
            if i == 0:
                label = tk.Label(self, text="f0(fund)", font=SMALL_FONT)
                label.grid(row=6, column=i + 1, padx=10, pady=5)
            else:
                label = tk.Label(self, text="f"+str(i), font=SMALL_FONT)
                label.grid(row=6, column=i+1, padx=10, pady=5)

        def checkBoxToggle():
            if customEnv.get():
                labelAttack.grid()
                labelDecay.grid()
                attackSc.grid()
                decaySc.grid()
                labelAttackValue.grid()
                labelDecayValue.grid()
                labels.grid()
                labels2.grid()
                labelSustainAmp.grid()
                sustainAmpSc.grid()
                labelSustainAmpValue.grid()
            else:
                labelAttack.grid_remove()
                labelDecay.grid_remove()
                attackSc.grid_remove()
                decaySc.grid_remove()
                labelAttackValue.grid_remove()
                labelDecayValue.grid_remove()
                labels.grid_remove()
                labels2.grid_remove()
                labelSustainAmp.grid_remove()
                sustainAmpSc.grid_remove()
                labelSustainAmpValue.grid_remove()

        customEnv = tk.BooleanVar()
        customEnv.set(1)
        radioEnv = tk.Checkbutton(self, text="custom the envelope", variable=customEnv, command=checkBoxToggle)
        radioEnv.grid(row=7, column=1, columnspan=6, pady=(40, 15))
        labelAttack = tk.Label(self, text="attack time", font=SMALL_FONT)
        labelAttack.grid(row=8, column=1, padx=10, pady=10)
        labelDecay = tk.Label(self, text="decay time", font=SMALL_FONT)
        labelDecay.grid(row=9, column=1, padx=10, pady=10)
        labelSustainAmp = tk.Label(self, text="sustain/attack\namplitude ratio", font=SMALL_FONT)
        labelSustainAmp.grid(row=10, column=1, padx=10, pady=0)
        attack_value = tk.DoubleVar()
        decay_value = tk.DoubleVar()
        sustain_ratio_value = tk.DoubleVar()
        attack_value.set(0.08)
        decay_value.set(0.05)
        sustain_ratio_value.set(0.4)
        attackSc = ttk.Scale(self, orient='horizontal', from_=0.01, to=0.5, variable=attack_value)
        attackSc.grid(row=8, column=2, columnspan=2, padx=10)
        decaySc = ttk.Scale(self, orient='horizontal', from_=0.01, to=0.5, variable=decay_value)
        decaySc.grid(row=9, column=2, columnspan=2, padx=10)
        sustainAmpSc = ttk.Scale(self, orient='horizontal', from_=0.01, to=1, variable=sustain_ratio_value)
        sustainAmpSc.grid(row=10, column=2, columnspan=2, padx=10)
        labelAttackValue = tk.Label(self, textvariable=attack_value, font=SMALL_FONT, width=5, anchor='w')
        labelAttackValue.grid(row=8, column=4, pady=10)
        labelDecayValue = tk.Label(self, textvariable=decay_value, font=SMALL_FONT, width=5, anchor='w')
        labelDecayValue.grid(row=9, column=4, pady=10)
        labelSustainAmpValue = tk.Label(self, textvariable=sustain_ratio_value, font=SMALL_FONT, width=5, anchor='w')
        labelSustainAmpValue.grid(row=10, column=4, pady=10)
        labels = tk.Label(self, text='sec', font=SMALL_FONT)
        labels.grid(row=8, column=4, padx=(0, 5), pady=10, sticky="e")
        labels2 = tk.Label(self, text='sec', font=SMALL_FONT)
        labels2.grid(row=9, column=4, padx=(0, 5), pady=10, sticky="e")

        def goCommand():
            if myaudio is None:
                popupmsg("ERROR : no audio file processed currently")
            else:
                amplitude_array = np.zeros(harm_number)
                inharmonicity_array = np.zeros(harm_number)
                for j in range(harm_number):
                    amplitude_array[j] = amplitudeVarList[j].get()
                    inharmonicity_array[j] = inHarmonicityVarList[j].get()
                attack = attack_value.get()
                decay = decay_value.get()
                sustainampratio = sustain_ratio_value.get()

                myaudio.customSynth(amplitude_array, inharmonicity_array, customEnv.get(), attack, decay, sustainampratio)

        goButton = ttk.Button(self, text='Go', command=goCommand)
        goButton.grid(row=12, column=0, columnspan=13, padx=12, pady=(80, 10), sticky="s")


# Creating and launching the app
app = TonalCompGui()
app.geometry("1280x720")
app.resizable(False, False)     # Some plots (like the stft) are a bit heavy and resizing the window can cause issues.
app.mainloop()
