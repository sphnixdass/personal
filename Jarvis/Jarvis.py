#!/usr/bin/env python3

import tkinter as tk
from tkinter import messagebox
import threading
import os
import subprocess
import time
import speech_recognition as sr
from gtts import gTTS
import requests
import json
import logging

# Global Variables
recording = False
process = None
# Setting up logging for better error tracking
logging.basicConfig(level=logging.INFO)

def listen():
    """
    Listens to the user's speech through the microphone and returns the recognized text.
    Handles different exceptions to provide specific error messages.
    """
    recognizer = sr.Recognizer()

    try:
        with sr.Microphone() as source:
            # Adjust for ambient noise to improve recognition accuracy
            recognizer.adjust_for_ambient_noise(source)
            logging.info("Listening for input...")

            # Listen for audio with a timeout of 15 seconds and a phrase time limit of 10 seconds
            audio = recognizer.listen(source, timeout=15, phrase_time_limit=10)

            # Use Google's speech recognition to convert audio to text
            recognized_text = recognizer.recognize_google(audio)
            logging.info(f"Recognized text: {recognized_text}")
            return recognized_text

    except sr.UnknownValueError:
        # Handle case where speech is not understood
        logging.error("Could not understand the audio.")
        return "Could not understand the audio."

    except sr.RequestError as e:
        # Handle case where there's a network or API issue with Google’s speech recognition service
        logging.error(f"Speech recognition service error: {e}")
        return f"Speech recognition service error: {e}"

    except sr.WaitTimeoutError:
        # Handle case where listening times out
        logging.error("Listening timed out.")
        return "Listening timed out."

    except Exception as e:
        # Catch any other unforeseen errors
        logging.error(f"An unexpected error occurred: {e}")
        return f"An unexpected error occurred: {e}"

def delete_mp3_files(directory):
    for filename in os.listdir(directory):
        if filename.endswith('.mp3'):
            os.remove(os.path.join(directory, filename))


def text_to_speech(text, output_file='output.mp3', speed=1.5):
    try:
        tts = gTTS(text=text, lang='en', slow=False)
        tts.save(output_file)
        os.system(f"ffmpeg -i {output_file} -af 'volume=2.0,atempo={speed}' temp_output.mp3")
        return subprocess.Popen(["ffplay", "-nodisp", "-loglevel", "quiet", "temp_output.mp3"])
    except Exception as e:
        return None


def generate_response(prompt):
    url = "http://localhost:11434/api/generate"
    headers = {"Content-Type": "application/json"}
    data = {"model": "jarvis", "prompt": prompt, "stream": False}
    response = requests.post(url, headers=headers, data=json.dumps(data))
    if response.status_code == 200:
        return response.json().get('response', '')
    return f"Error: {response.status_code}"


def toggle_record():
    global recording, process
    if recording:
        recording = False
        record_button.config(text="Record")
    else:
        recording = True
        record_button.config(text="Recording...")
        threading.Thread(target=record_loop, daemon=True).start()


def record_loop():
    global process
    while recording:
        user_input = listen()
        if user_input:
            speech_to_text_box.delete(1.0, tk.END)
            speech_to_text_box.insert(tk.END, user_input)
            user_input_code = user_input_box.get(1.0, tk.END).strip()

            if user_input.lower() in ["exit", "quit"]:
                messagebox.showinfo("Exiting", "Goodbye!")
                app.quit()
                return

            delete_mp3_files("./")
            jarvis_output = generate_response(user_input + "\n=======\n" + user_input_code)
            jarvis_output_box.delete(1.0, tk.END)
            jarvis_output_box.insert(tk.END, jarvis_output)

            if process:
                process.terminate()
            process = text_to_speech(jarvis_output)


def stop_audio():
    global process
    if process:
        process.terminate()
        process = None
        print("Playback stopped.")


def send_user_input():

    global process
 

    user_input_code = user_input_box.get(1.0, tk.END).strip()
    user_input = speech_to_text_box.get(1.0, tk.END).strip()


    delete_mp3_files("./")
    jarvis_output = generate_response(user_input + "\n=======\n" + user_input_code)
    jarvis_output_box.delete(1.0, tk.END)
    jarvis_output_box.insert(tk.END, jarvis_output)

    if process:
        process.terminate()
    process = text_to_speech(jarvis_output)


    # global process
    # user_input = user_input_box.get(1.0, tk.END).strip()
    # if user_input:
    #     delete_mp3_files("./")
    #     jarvis_output = generate_response(user_input)
    #     jarvis_output_box.delete(1.0, tk.END)
    #     jarvis_output_box.insert(tk.END, jarvis_output)

    #     if process:
    #         process.terminate()
    #     process = text_to_speech(jarvis_output)


def exit_app():
    stop_audio()
    app.quit()

# Create GUI
app = tk.Tk()
app.title("Jarvis")

app.geometry("500x600")
# Frame for buttons to be displayed in a single row
button_frame = tk.Frame(app)
button_frame.pack(pady=10)

# Buttons in the button frame
record_button = tk.Button(button_frame, text="Record", command=toggle_record, width=15)
record_button.pack(side=tk.LEFT, padx=5)

stop_button = tk.Button(button_frame, text="Stop", command=stop_audio, width=15)
stop_button.pack(side=tk.LEFT, padx=5)

exit_button = tk.Button(button_frame, text="Exit", command=exit_app, width=15)
exit_button.pack(side=tk.LEFT, padx=5)

# Function to add text box with a scrollbar
def add_textbox_with_scrollbar(parent, height, width):
    frame = tk.Frame(parent)
    frame.pack(pady=5)

    textbox = tk.Text(frame, height=height, width=width)
    textbox.pack(side=tk.LEFT)

    scrollbar = tk.Scrollbar(frame, command=textbox.yview)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    textbox.config(yscrollcommand=scrollbar.set)
    return textbox

# Textboxes with scrollbars
speech_to_text_label = tk.Label(app, text="Speech-to-Text Data")
speech_to_text_label.pack()

speech_to_text_box = add_textbox_with_scrollbar(app, height=5, width=60)

jarvis_output_label = tk.Label(app, text="Jarvis Output Data")
jarvis_output_label.pack()

jarvis_output_box = add_textbox_with_scrollbar(app, height=15, width=60)

user_input_label = tk.Label(app, text="User Input Text")
user_input_label.pack()

user_input_box = add_textbox_with_scrollbar(app, height=2, width=60)

# Send button
send_button = tk.Button(app, text="Send", command=send_user_input, width=15)
send_button.pack(pady=10)

# Start the app
app.mainloop()
