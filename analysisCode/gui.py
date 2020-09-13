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


def popupmsg(msg):
    popup = tk.Tk()
    popup.wm_title("!")
    label = ttk.Label(popup, text=msg, font=NORM_FONT)
    label.pack(side="top", fill="x", pady=10)
    b1 = ttk.Button(popup, text="Okay", command=popup.destroy)
    b1.pack()
    popup.mainloop()


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


def show_peaks(tk_win, num_value):
    tk_win.destroy()
    try:
        value = int(num_value)
    except ValueError:
        popupmsg("Invalid number format")
        return

    if 1 <= value <= 10:
        myaudio.showpeaks(value)
    else:
        popupmsg("Selected number out of range (1 to 10)")


def ask_peaks_num():
    if myaudio is None:
        popupmsg("ERROR : no audio file processed currently")
    else:
        num = tk.Tk()
        num.wm_title("Peak number selection")
        label = ttk.Label(num, text=f"select your number of peaks from 1 to 10", font=NORM_FONT)
        label.pack(side="top", fill="x", pady=10)

        num_value = ttk.Entry(num)
        num_value.pack()

        b1 = ttk.Button(num, text="Ok", command=lambda: show_peaks(num, num_value.get()))
        b1.pack()
        num.mainloop()



class TonalCompGui(tk.Tk):

    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)

        # tk.Tk.iconbitmap(self, default="clienticon.ico")
        tk.Tk.wm_title(self, "TonalComp")

        container = tk.Frame(self)
        container.pack(side="top", fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        menubar = tk.Menu(container)
        filemenu = tk.Menu(menubar, tearoff=0)
        filemenu.add_command(label="Back to lobby", command=lambda: self.show_frame(StartPage))
        filemenu.add_separator()
        filemenu.add_command(label="Exit", command=self.destroy) #"command = quit" causes issues
        menubar.add_cascade(label="File", menu=filemenu)

        analysismenu = tk.Menu(menubar, tearoff=0)
        analysismenu.add_command(label="show frame", command=ask_frame_num)
        analysismenu.add_command(label="show peaks", command=ask_peaks_num)
        analysismenu.add_command(label="show final trajectories", command=lambda: myaudio.show_trajectories())
        menubar.add_cascade(label="Analysis", menu=analysismenu)

        synthmenu = tk.Menu(menubar, tearoff=0)
        synthmenu.add_command(label="re-synthesis", command=lambda: popupmsg("not implemented yet"))
        synthmenu.add_command(label="custom synthesis", command=lambda: self.show_frame(CustomSynthesisPage))
        menubar.add_cascade(label="Synthesis", menu=synthmenu)

        tk.Tk.config(self, menu=menubar)

        self.frames = {}

        for F in (StartPage, LoadingPage, MainPage, CustomSynthesisPage):
            frame = F(container, self)

            self.frames[F] = frame

            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame(StartPage)

    def show_frame(self, cont):
        if cont == MainPage:    # This is the wrong place to do it but I didn't find better yet
            self.frames[cont].infosText.set(myaudio.pathName+"\n\nFs: "+str(myaudio.Fs)+"\nN_fft: "+str(myaudio.N_fft) +
                                            "\nN_frames: "+str(myaudio.n_frames))
        frame = self.frames[cont]
        frame.tkraise()


class StartPage(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.grid_columnconfigure(0, weight=1)
        label = tk.Label(self, text=("Hi and welcome to TonalComp. \nSelect an audio file, "
                                     "or pick an example"), font=LARGE_FONT)
        label.grid(row=0, pady=30, sticky="nsew")

        listbox = tk.Listbox(self)
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
        button1.grid(row=2)

        self.labelText = tk.StringVar()
        self.labelText.set("no audio file selected")
        label2 = tk.Label(self, textvariable=self.labelText, font=NORM_FONT)
        label2.grid(row=3, padx=20, sticky="nsew")

        self.paramBg = "grey70"
        parametersFrame = tk.Frame(self, height=150, background=self.paramBg)
        parametersFrame.grid(row=4, padx=100, pady=50)

        tk.Label(parametersFrame, text="Analysis parameters", font=LARGE_FONT, background=self.paramBg).grid(row=0,
                                                                                        sticky="nsew", padx=15, pady=15)

        tk.Label(parametersFrame, text="win_length multiplicator (>1)", background=self.paramBg).grid(row=1, sticky="e")
        tk.Label(parametersFrame, text="N_fft multiplicator", background=self.paramBg).grid(row=2, sticky="e")

        self.winLength_mul_str = tk.StringVar()
        self.nfft_mul_str = tk.StringVar()
        self.fmin_str = tk.StringVar()
        self.fmax_str = tk.StringVar()

        self.winLength_mul_str.set("1")
        # choices = ['0.25', '0.5', '1', '2', '4']
        choices = ['1']
        self.nfft_mul_str.set('1')
        self.fmin_str.set("50")
        self.fmax_str.set("20000")
        e1 = tk.Entry(parametersFrame, textvariable=self.winLength_mul_str)
        e2 = tk.OptionMenu(parametersFrame, self.nfft_mul_str, *choices)
        e1.grid(row=1, column=1, sticky="w", padx=(5, 20))
        e2.grid(row=2, column=1, sticky="w", padx=(5, 20))

        tk.Label(parametersFrame, text="Bandwidth", font=NORM_FONT, background=self.paramBg).grid(row=3, sticky="e", pady=(10, 0))

        tk.Label(parametersFrame, text="f_min", background=self.paramBg).grid(row=4, sticky="e")
        tk.Label(parametersFrame, text="f_max (no space)", background=self.paramBg).grid(row=5, sticky="e", pady=(0, 20))

        e3 = tk.Entry(parametersFrame, textvariable=self.fmin_str)
        e4 = tk.Entry(parametersFrame, textvariable=self.fmax_str)
        e3.grid(row=4, column=1, sticky="w", padx=(5, 20))
        e4.grid(row=5, column=1, sticky="w", padx=(5, 20), pady=(0, 20))

        buttonMore = ttk.Button(parametersFrame, text="Learn more about these parameters")
        buttonMore.grid(row=6, column=0, columnspan=2)

        def goAction():
            controller.show_frame(LoadingPage)
            controller.update()
            global myaudio
            # app.config(cursor="wait") # not working
            time_1 = time.time()
            try:
                print(float(self.winLength_mul_str.get()))
                print(float(self.nfft_mul_str.get()))
                analysisParams = audioAnalysis.AnalysisParameters(int(self.fmin_str.get()), int(self.fmax_str.get()))
                myaudio = audioAnalysis.AudioAnalysis(self.selectedPath, analysisParams, float(self.winLength_mul_str.get()), float(self.nfft_mul_str.get()))
            except ValueError:
                controller.show_frame(StartPage)
                popupmsg("ERROR : Analysis parameters entry format error")

            print(f"\n\nComputation in {time.time()-time_1} seconds")
            time_2 = time.time()
            librosa_display_subplt.clear()
            librosa.display.specshow(myaudio.X_db, y_axis='log', x_axis='time', ax=librosa_display_subplt)
            plt.title('Frequency spectrogram')  # not working
            print(f"\n\nLibrosa display computation in {time.time()-time_2} seconds")
            time_3 = time.time()
            f.canvas.draw()
            f.canvas.flush_events()
            # app.config(cursor="")     # not working
            controller.show_frame(MainPage)
            print(f"\n\nCanvas drawing in {time.time()-time_3} seconds")

        self.button3 = ttk.Button(self, text="GO", width=50, state=tk.DISABLED, command=goAction)
        self.button3.grid(row=5, padx=50, pady=50)

    def selectFile(self):
        filename = tk.filedialog.askopenfilename(filetypes=(("Audio files (wav,m4a)", "*.wav;*.m4a")
                                                ,("All files", "*.*")))  # mp3 not supported yet (can be with ffmpeg)
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


class LoadingPage(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="Loading...", font=LARGE_FONT)
        label.pack(pady=10, padx=10)


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


class CustomSynthesisPage(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)

        titleLabel = tk.Label(self, text="CustomSynthesis", font=LARGE_FONT)
        titleLabel.grid(row=0, column=0, columnspan=12, sticky='nsew')

        backButton = ttk.Button(self, text='Back', command=lambda: controller.show_frame(MainPage))
        backButton.grid(row=1, column=0, padx=10)

        subtitleLabel = tk.Label(self, text="custom your harmonics", font=NORM_FONT)
        subtitleLabel.grid(row=1, column=3, columnspan=6, pady=20)

        labelAmplitude = tk.Label(self, text="Amplitude", font=SMALL_FONT)
        labelAmplitude.grid(row=3, column=0, padx=10, pady=10)
        labelInharmo = tk.Label(self, text="Inharmonicity", font=SMALL_FONT)
        labelInharmo.grid(row=4, column=0, padx=10, pady=10)

        amplitudeVarList = []
        inHarmonicityVarList = []
        for i in range(12):
            amplitudeVarList.append(tk.DoubleVar())
            inHarmonicityVarList.append(tk.DoubleVar())
            amplitudeVarList[i].set(1/(i+1))
            inHarmonicityVarList[i].set(0)

            harmoScale = ttk.Scale(self, orient='vertical', from_=1, to=0, variable=amplitudeVarList[i])
            harmoScale.grid(row=3, column=i+1, padx=10)
            labelValue = tk.Label(self, textvariable=amplitudeVarList[i], font=SMALL_FONT, width=5, anchor='w')
            labelValue.grid(row=3, column=i+1, padx=15)

            inharmonicityScale = ttk.Scale(self, orient='horizontal', from_=-1, to=1, length=50, variable=inHarmonicityVarList[i])
            inharmonicityScale.grid(row=4, column=i+1, padx=10)
            labelValue = tk.Label(self, textvariable=inHarmonicityVarList[i], font=SMALL_FONT, width=4, anchor='w')
            labelValue.grid(row=5, column=i+1, padx=10)
            if i == 0:
                label = tk.Label(self, text="fund", font=SMALL_FONT)
                label.grid(row=6, column=i + 1, padx=10)
            else:
                label = tk.Label(self, text="f"+str(i), font=SMALL_FONT)
                label.grid(row=6, column=i+1, padx=10)

        envelopeLabel = tk.Label(self, text="custom the envelope", font=NORM_FONT)
        envelopeLabel.grid(row=7, column=1, columnspan=6, pady=(30,15))
        labelAttack = tk.Label(self, text="attack time", font=SMALL_FONT)
        labelAttack.grid(row=8, column=1, padx=10, pady=10)
        labelDecay = tk.Label(self, text="decay time", font=SMALL_FONT)
        labelDecay.grid(row=9, column=1, padx=10, pady=10)
        attackSc = ttk.Scale(self, orient='horizontal', from_=0, to=1)
        attackSc.grid(row=8, column=2, padx=10)
        decaySc = ttk.Scale(self, orient='horizontal', from_=0, to=1)
        decaySc.grid(row=9, column=2, padx=10)

        goButton = ttk.Button(self, text='Go', command=None)
        goButton.grid(row=12, column=0, padx=12)



app = TonalCompGui()
app.geometry("1280x720")
app.mainloop()
