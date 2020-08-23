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
myaudio = None


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
        menubar.add_cascade(label="Analysis", menu=analysismenu)

        synthmenu = tk.Menu(menubar, tearoff=0)
        synthmenu.add_command(label="do smth", command=lambda: popupmsg("not implemented yet"))
        menubar.add_cascade(label="Synthesis", menu=synthmenu)

        tk.Tk.config(self, menu=menubar)

        self.frames = {}

        for F in (StartPage, MainPage):
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
        label = tk.Label(self, text=("Hi and welcome to TonalComp. \nSelect an audio file, "
                                     "or pick an example"), font=LARGE_FONT)
        label.pack(pady=10, padx=10)

        listbox = tk.Listbox(self)
        listbox.pack(side=tk.TOP, fill='x', padx=50, pady=50)

        self.selectedPath = ""

        for file in glob.glob("..\\demo_sound\\*.wav"):
            listbox.insert(-1, file)

        def selectItem(evt):
            self.selectedPath = listbox.get(listbox.curselection())
            self.labelText.set("Audio file : " + self.selectedPath)
            self.button3['state'] = 'normal'
        listbox.bind("<<ListboxSelect>>", selectItem)

        button1 = ttk.Button(self, text="Select your own file file", command=lambda: self.selectFile())
        button1.pack()

        self.labelText = tk.StringVar()
        self.labelText.set("no audio file selected")
        label2 = tk.Label(self, textvariable=self.labelText, font=NORM_FONT)
        label2.pack(pady=20, padx=20)

        def goAction():
            global myaudio
            # app.config(cursor="wait") # not working
            analysisParams = audioAnalysis.AnalysisParameters()
            myaudio = audioAnalysis.AudioAnalysis(self.selectedPath, analysisParams)
            librosa_display_subplt.clear()
            librosa.display.specshow(myaudio.X_db, y_axis='log', x_axis='time', ax=librosa_display_subplt)
            plt.title('Frequency spectrogram')  # not working
            f.canvas.draw()
            f.canvas.flush_events()
            # app.config(cursor="")     # not working
            controller.show_frame(MainPage)

        self.button3 = ttk.Button(self, text="GO", state=tk.DISABLED, command=goAction)
        self.button3.pack(side="bottom", pady=50)

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


app = TonalCompGui()
app.geometry("1280x720")
app.mainloop()
