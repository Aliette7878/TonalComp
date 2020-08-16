# The code for changing pages was derived from: http://stackoverflow.com/questions/7546050/switch-between-two-frames-in-tkinter
# Code addapted from tutorial here : https://pythonprogramming.net/tkinter-depth-tutorial-making-actual-program/
# License: http://creativecommons.org/licenses/by-sa/3.0/

import matplotlib

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib import style

import tkinter as tk
from tkinter import ttk

import glob

from matplotlib import pyplot as plt

import librosa.display
import main         # importing script main.py


LARGE_FONT = ("Verdana", 12)
NORM_FONT = ("Verdana", 10)
SMALL_FONT = ("Verdana", 8)

matplotlib.use("TkAgg")
style.use("ggplot")

f = Figure()
a = f.add_subplot(111)
librosa.display.specshow(main.X_db, y_axis='log', x_axis='time', ax=a)
plt.title('Frequency spectrogram')
plt.tight_layout()


def popupmsg(msg):
    popup = tk.Tk()
    popup.wm_title("!")
    label = ttk.Label(popup, text=msg, font=NORM_FONT)
    label.pack(side="top", fill="x", pady=10)
    b1 = ttk.Button(popup, text="Okay", command=popup.destroy)
    b1.pack()
    popup.mainloop()


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
        filemenu.add_command(label="show smth", command=lambda: popupmsg("Not supported just yet!"))
        filemenu.add_separator()
        filemenu.add_command(label="Exit", command=quit)
        menubar.add_cascade(label="File", menu=filemenu)

        tk.Tk.config(self, menu=menubar)

        self.frames = {}

        for F in (StartPage, MainPage):
            frame = F(container, self)

            self.frames[F] = frame

            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame(StartPage)

    def show_frame(self, cont):
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

        for file in glob.glob("..\\demo_sound\\*.wav"):
            listbox.insert(-1, file)

        def selectItem(evt):
            labelText.set("Audio file : "+listbox.get(listbox.curselection()))
            button3['state'] = 'normal'
        listbox.bind("<<ListboxSelect>>", selectItem)

        button1 = ttk.Button(self, text="Select your own file file", command=lambda: popupmsg("Not supported yet"))
        button1.pack()

        labelText = tk.StringVar()
        labelText.set("no audio file selected")
        label2 = tk.Label(self, textvariable=labelText, font=NORM_FONT)
        label2.pack(pady=10, padx=10)

        button3 = ttk.Button(self, text="GO", state=tk.DISABLED, command=lambda: controller.show_frame(MainPage))
        button3.pack(side="bottom", pady=50)


class MainPage(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="Main Page", font=LARGE_FONT)
        label.pack(pady=10, padx=10)

        lobbyButton = ttk.Button(self, text="Back to Lobby", command=lambda: controller.show_frame(StartPage))
        lobbyButton.pack()

        # We need to set a canva to be able to integrate any matplotlib printing in our frame
        canvas = FigureCanvasTkAgg(f, self)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2Tk(canvas, self)
        toolbar.update()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)


app = TonalCompGui()
app.geometry("1280x720")
app.mainloop()
