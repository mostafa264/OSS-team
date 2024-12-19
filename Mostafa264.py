from tkinter import *
from tkinter import filedialog
import tkinter as tk
import subprocess
import os
# from PIL import Image,ImageTk
from stegano import *

def show_image():
    global filename
    filename=filedialog.askopenfilename(initialdir=os.getcwd(),
                                    title="Select Image File",
                                    filetypes=(("PNG file","*.png"),
                                               ("JPG file","*.jpg"),
                                               ("All file","*.txt")))
    img=Image.open(filename)
    img=ImageTk.PhotoImage(img)
    lb1.configure(image=img,width=250,height=250)
    lb1.image=img

def hide():
    global secret
    message=txt1.get(1.0,END)
    secret=lsb.hide(str(filename),message)

def extract():
    clear_message =lsb.reveal(filename)
    txt1.delete(1.0,END)
    txt1.insert(END,clear_message)

def save():
    secret.save("hidden.png")

def gif_window():
    global lb1
    global txt1
    new_window = tk.Toplevel(root)
    new_window.title("GIF TOOL")
    new_window.geometry("700x500")
    new_window.resizable(False, False)  # user can't resize window.
    new_window.configure(bg="#0e3458")

    # First Frame
    f= Frame(new_window,bd=3,bg="black",width=300,height=270,relief=GROOVE)
    f.place(x=20,y=80)
    lb1=Label(f,bg="black")
    lb1.place(x=40,y=10)

    # Second Frame
    f2= Frame(new_window,bd=3,bg="white",width=300,height=270,relief=GROOVE)
    f2.place(x=360,y=80)
    txt1=Text(f2, font="arial 20", bg="white", fg="black", relief=GROOVE, wrap=WORD)
    txt1.place(x=0,y=0,width=330, height= 280)

    # Third Frame
    f3= Frame(new_window,bd=3,bg="#0e3458",width=300,height=100,relief=GROOVE)
    f3.place(x=20,y=350)
    Button(f3,text="Open", width=8,height=2, font="arial 15 bold",command=show_image).place(x=20,y=18)
    Button(f3,text="Save", width=8,height=2, font="arial 15 bold", command=save).place(x=150,y=18)
    Label(f3, text="Image File", bg="#0e3458", fg="white", font="arial 7 bold").place(x=20,y=0)

    # Fourth Frame
    f4 = Frame(new_window, bd=3, bg="#0e3458", width=300, height=100, relief=GROOVE)
    f4.place(x=360, y=350)
    Button(f4, text="Hide", width=8, height=2, font="arial 15 bold",command=hide).place(x=20, y=18)
    Button(f4, text="Extract", width=8, height=2, font="arial 15 bold",command=extract).place(x=150, y=18)
    Label(f4, text="Choose Operaation", bg="#0e3458", fg="white", font="arial 7 bold").place(x=20, y=0)


root=Tk()
title_label= root.title("Steganography Project")
root.geometry("700x500")
root.resizable(False,False) # user can't resize window.
root.configure(bg="#0e3458")

# LABEL
Label(root, text="Stego Sheild", bg="#0e3458", fg="white", font="arial 28 bold").place(x=250, y=20)

# Functions
def run_image_exe():
    try:
        subprocess.run(["D:\Downloud\data\data1\collage\the forth year\Steganography\Sections\Section 4\Tools\S-Tools\s-tools4\S-Tools.exe"], check=True)
    except Exception as i:
        print(f"Error: {i}")

def run_audio_exe():
    try:
        subprocess.run(["C:\ProgramData\Microsoft\Windows\Start Menu\Programs\DeepSound 2.0\DeepSound.lnk"], check=True)
    except Exception as e:
        print(f"Error: {e}")

def run_video_exe():
    try:
        subprocess.run(["C:\ProgramData\Microsoft\Windows\Start Menu\Programs\DeEgger Embedder\DeEgger Embedder.lnk"], check=True)
    except Exception as v:
        print(f"Error: {v}")

def run_gif_exe():
    try:
        subprocess.run(["C:\ProgramData\Microsoft\Windows\Start Menu\Programs\DeEgger Embedder\DeEgger Embedder.lnk"], check=True)
    except Exception as g:
        print(f"Error: {g}")
# Image Button
i_button = tk.Button(
    root,
    text="Image Steganography",
    relief="flat",
    bg="#67cbee",
    activebackground="#5B6EAE",
    fg="white",
    font=("Arial", 14, "bold"),
    command=run_image_exe
)
i_button.pack(pady=50)
i_button.place(x=80,y=100)

# Audio Button
a_button = tk.Button(
    root,
    text="Audio Steganography",
    relief="flat",
    bg="#67cbee",
    activebackground="#5B6EAE",
    fg="white",
    font=("Arial", 14, "bold"),
    command=run_audio_exe
)
a_button.pack(pady=50)
a_button.place(x=400,y=100)

# Video Button
v_button = tk.Button(
    root,
    text="Video Steganography",
    relief="flat",
    bg="#67cbee",
    activebackground="#5B6EAE",
    fg="white",
    font=("Arial", 14, "bold"),
    command=run_video_exe
)
v_button.pack(pady=50)
v_button.place(x=400,y=200)

# Gif Button
g_button = tk.Button(
    root,
    text="GIF Tool",
    relief="flat",
    bg="#67cbee",
    activebackground="#5B6EAE",
    fg="white",
    font=("Arial", 14, "bold"),
    command= gif_window
)
g_button.pack(pady=50)
g_button.place(x=80,y=200)
root.mainloop()

