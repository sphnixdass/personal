from flask import Flask, request, jsonify, render_template, Response
import requests
import json
from flask_executor import Executor
from time import sleep
import speech_recognition as sr

# import tkinter as tk
# from tkinter import messagebox
import threading
import os
import subprocess
import time
import speech_recognition as sr
from gtts import gTTS
import requests
import json
import logging

import pyaudio
import wave
import numpy as np


app = Flask(__name__,static_url_path='/static')
executor = Executor(app)

# Global Variables
recording = False
process = None
strmp3= ""
# Setting up logging for better error tracking
logging.basicConfig(level=logging.INFO)
recognizer = sr.Recognizer()
# audio = pyaudio.PyAudio()


def convert_wav_to_text(wav_file):
    """Convert a .wav file to text using SpeechRecognition."""
    # Initialize the recognizer
    # recognizer = sr.Recognizer()

    # Load the audio file
    with sr.AudioFile(wav_file) as source:
        print("Loading audio file...")
        audio_data = recognizer.record(source)  # Read the entire audio file

    # Perform speech-to-text
    try:
        print("Transcribing audio...")
        text = recognizer.recognize_google(audio_data)  # Use Google's Web Speech API
        print("Transcription completed!")
        if text == "":
            return None
        return text
    except sr.UnknownValueError:
        print("Speech was unintelligible.")
        return None
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")
        return None

# Audio settings
FORMAT = pyaudio.paInt16  # Audio format
CHANNELS = 1             # Mono audio
RATE = 44100             # Sample rate
CHUNK = 1024             # Buffer size
THRESHOLD = 500          # Silence threshold (lower values are more sensitive)
SILENCE_DURATION = 2     # Silence duration to stop recording (in seconds)
MAX_WAIT_TIME = 60       # Maximum recording time (in seconds)

def is_silent(data, threshold=THRESHOLD):
    """Check if the audio data is below the silence threshold."""
    audio_data = np.frombuffer(data, dtype=np.int16)
    return np.max(np.abs(audio_data)) < threshold

def record_until_silent(output_filename="output.wav"):
    """Record audio until silence or maximum wait time is reached, and save as a .wav file."""
    audio = pyaudio.PyAudio()
    print("Opening audio stream...")
    # Open stream
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        frames_per_buffer=CHUNK)

    print("Recording...")

    frames = []
    silent_chunks = 0
    total_chunks = 0
    silence_limit = int(RATE / CHUNK * SILENCE_DURATION)
    max_chunks = int(RATE / CHUNK * MAX_WAIT_TIME)

    try:
        while True:
            data = stream.read(CHUNK, exception_on_overflow=False)
            frames.append(data)
            total_chunks += 1

            if is_silent(data):
                silent_chunks += 1
            else:
                silent_chunks = 0

            # Stop if silence exceeds limit or if max wait time is reached
            if silent_chunks > silence_limit:
                print("Silence detected, stopping recording.")
                break
            if total_chunks > max_chunks:
                print("Maximum wait time reached, stopping recording.")
                break
    except KeyboardInterrupt:
        print("Recording interrupted by user.")

    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    audio.terminate()

    # Save the recorded audio as a .wav file
    with wave.open(output_filename, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b"".join(frames))

    print(f"Recording saved to {output_filename}")

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



def stop_audio():
    global process
    print("Stopping playback...")
    if process:
        process.terminate()
        process = None
        print("Playback stopped.")


def send_user_input(input_text):

    global process
 
    delete_mp3_files("./")

    if process:
        process.terminate()
    process = text_to_speech(input_text)



def exit_app():
    stop_audio()
    app.quit()



@app.route('/')
def index():
    return render_template('index.html')

# @executor.job
def long_running_task():
    global strmp3
    global process
 
    """
    Simulates a long-running task.
    """
    print("Long-running task started.")
    delete_mp3_files("./")
    if process:
        process.terminate()
    process = text_to_speech(strmp3)

    # text_to_speech(strmp3)

    # print("Starting long-running task...")
    # sleep(10)  # Simulate a 10-second task
    print("Long-running task completed.")
    # return "Task finished successfully"



def generate_json(user_messages, assistant_responses):
    data = {
        "model": "llama3.2",
        "messages": []
    }
    
    for user_message, assistant_response in zip(user_messages, assistant_responses):
        data["messages"].append({
            "role": "user",
            "content": user_message
        })
        data["messages"].append({
            "role": "assistant",
            "content": assistant_response
        })
    
    # Add the last user message without an assistant response
    if len(user_messages) > len(assistant_responses):
        data["messages"].append({
            "role": "user",
            "content": user_messages[-1]
        })
    
    data["stream"] = False
    return json.dumps(data, indent=4)

# Example user messages and assistant responses
user_messages = []

assistant_responses = [
]


def speachloop():
    global strmp3
    mictext = ""
   
    # print("Recording...")
    record_until_silent("recording.wav")
    mictext = convert_wav_to_text("recording.wav")


    print("Chat voice function called", mictext)
    if mictext == None:
        print("No input detected, please try again.")
    else:
        mictext = str(mictext)

    mictext = str(mictext)
    if (mictext.strip() != "" and mictext != "None"):
        user_message = mictext
        user_messages.append(user_message)
        # Generate and print the JSON data
        json_data = generate_json(user_messages, assistant_responses)
        print(json_data)



        headers = {'Content-Type': 'application/json'}
        response = requests.post('http://127.0.0.1:11434/api/chat', json=json.loads(json_data), headers=headers, stream=True)
        user_messages.append(user_message)
        jarvis_output = str(json.loads(response.text)['message']['content'])
        assistant_responses.append(jarvis_output)
        strmp3 = jarvis_output
        # executor.submit(long_running_task) # Start the long-running task
        long_running_task()

        print("Response:", jarvis_output)

        # lines = str(jarvis_output).splitlines()
        # tempData = ""
        # for line in lines:
        #     tempData += f"<p>{line}</p>"
        
        # return tempData
        return jarvis_output + "<Dass>" + user_message
    else:
        return ""
        

@app.route('/clear')
def clear():
    user_messages.clear()
    assistant_responses.clear()
    return "Messages cleared", 200

@app.route('/stopAudio')
def stopAudio():
    stop_audio()
    return "Playback stopped", 200

@app.route('/playmp3')
def playmp3():
    user_messages.clear()
    assistant_responses.clear()
    return "Messages cleared", 200

@app.route('/speechInput')
def speechInput():
    return Response(speachloop(), mimetype='text/html')


@app.route('/chat', methods=['POST'])
def chat():
    global strmp3
    print("Chat function called", request.form['userMessage'])
    user_message = request.form['userMessage']
    user_messages.append(user_message)
    # Generate and print the JSON data
    json_data = generate_json(user_messages, assistant_responses)
    print(json_data)



    headers = {'Content-Type': 'application/json'}
    response = requests.post('http://127.0.0.1:11434/api/chat', json=json.loads(json_data), headers=headers, stream=True)
    user_messages.append(user_message)
    jarvis_output = str(json.loads(response.text)['message']['content'])
    assistant_responses.append(jarvis_output)
    strmp3 = jarvis_output
    # executor.submit(long_running_task) # Start the long-running task
    long_running_task()

    def generate():
        print("Response:", jarvis_output)

        # lines = str(jarvis_output).splitlines()
        # tempData = ""
        # for line in lines:
        #     tempData += f"<p>{line}</p>"
         
        # return tempData
        return jarvis_output
    
    # return "Hello", 200
    return Response(generate(), mimetype='text/html')
    # return Response(json.loads(response.text)['message']['content'], mimetype='text/html')

if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=True, port=3010)