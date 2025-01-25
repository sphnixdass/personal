from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
import os
import signal

from selenium.webdriver.common.by import By
from tkinter import Tk, Button, Label, Scale, HORIZONTAL, DISABLED, NORMAL
import threading
import time
import requests

# API endpoints
CONVERT_API_URL = "http://127.0.0.1:8000/convert-text-to-mp3/"  # Update with your convert API URL
PLAY_API_URL = "http://127.0.0.1:8000/play-mp3/"
PAUSE_API_URL = "http://127.0.0.1:8000/pause-mp3/"
STOP_API_URL = "http://127.0.0.1:8000/stop-mp3/"

# Selenium setup
driver = None
paragraphs = []
current_index = 0
is_running = False
is_paused = False

# GUI buttons
start_button = None
pause_button = None
stop_button = None
speed_scale = None

# Start the Selenium browser for user navigation
def open_browser():
    global driver

    try:

        # Set up Chrome options
        chrome_options = Options()
        chrome_options.debugger_address = "127.0.0.1:9222"

        # Path to ChromeDriver
        service = Service("/usr/bin/chromedriver")  # Adjust path as needed

        # Connect to Chrome
        driver = webdriver.Chrome(service=service, options=chrome_options)


        # service = Service("geckodriver")  # Path to GeckoDriver
        # driver = webdriver.Firefox(service=service)
        driver.maximize_window()
        print("Browser launched. Navigate to your desired website.")
    except Exception as e:
        print(f"Error launching browser: {e}")

# Fetch data from the current browser page
def fetch_data():
    global paragraphs, current_index
    if not driver:
        print("Browser is not open. Please launch the browser first.")
        return

    try:
        paragraphs = driver.find_elements(By.TAG_NAME, "p")  # Collect <p> elements
        current_index = 0  # Reset the index
        print(f"Fetched {len(paragraphs)} paragraphs from the current page.")

        # Enable control buttons after fetching data
        start_button.config(state=NORMAL)
        pause_button.config(state=NORMAL)
        stop_button.config(state=NORMAL)
    except Exception as e:
        print(f"Error fetching data: {e}")

# Function to send text to Convert TTS API
def send_to_convert_api(text, argcurrent_index):
    try:
        response = requests.post(CONVERT_API_URL, json={"text": text, "lang": "en", "argcurrent_index": argcurrent_index})
        if response.status_code == 200:
            print(f"Conversion Successful: {response.json().get('message', 'Success')}")
        else:
            print(f"Conversion Failed: {response.text}")
    except Exception as e:
        print(f"Error: {e}")

# Function to send play, pause, or stop commands to API
def send_audio_control_api(api_url, data=None):
    try:
        response = requests.post(api_url, json=data)
        if response.status_code == 200:
            print(f"Command Successful: {response.json().get('message', 'Success')}")
        else:
            print(f"Command Failed: {response.text}")
    except Exception as e:
        print(f"Error: {e}")

# Highlight the current paragraph
def highlight_paragraph(index):
    script = """
    arguments[0].style.border = '2px solid red';
    arguments[0].style.backgroundColor = 'black';
    arguments[0].style.color = 'green';
    arguments[0].scrollIntoView({behavior: 'smooth', block: 'center'});
    """
    driver.execute_script(script, paragraphs[index])

# Remove highlight from the previous paragraph
def remove_highlight(index):
    script = """
    arguments[0].style.border = '';
    arguments[0].style.backgroundColor = '';
    """
    driver.execute_script(script, paragraphs[index])

# Function to check if MP3 is playing
def is_mp3_playing():
    try:
        response = requests.get("http://127.0.0.1:8000/is-playing-mp3/")  # Update with the correct endpoint
        if response.status_code == 200:
            return response.json().get("is_playing", False)
    except Exception as e:
        print(f"Error checking playback status: {e}")
    return False

# Thread for reading paragraphs and sending to Convert API
def process_paragraphs():
    global current_index, is_running, is_paused

    next_conversion_thread = None

    while is_running and current_index < len(paragraphs):
        if is_paused:
            time.sleep(1)  # Wait while paused
            continue

        # Highlight current paragraph
        highlight_paragraph(current_index)

        # Process text
        text = paragraphs[current_index].text.strip()
        if text:
            print(f"Processing paragraph {current_index + 1}/{len(paragraphs)}: {text}")

            # Start conversion for the next paragraph in a separate thread
            # if next_conversion_thread:
            #     next_conversion_thread.join()  # Ensure the previous thread has completed
            # next_conversion_thread = threading.Thread(target=send_to_convert_api, args=(text, current_index))
            # next_conversion_thread.start()

            send_to_convert_api(text, current_index)

        # Play the converted MP3 with the selected speed
        speed = speed_scale.get()
        time.sleep(1)
        send_audio_control_api(PLAY_API_URL, {"speed": speed})
        time.sleep(1)
        # Wait for MP3 playback to complete
        while is_mp3_playing():
            time.sleep(1)

        # Remove highlight after processing
        # time.sleep(2)  # Wait for a short period to highlight
        remove_highlight(current_index)

        # Move to the next paragraph
        current_index += 1
        # time.sleep(1)  # Delay between processing paragraphs

    if current_index >= len(paragraphs):
        print("All paragraphs processed.")
        is_running = False

# GUI functions
def fetch_data_button_action():
    threading.Thread(target=fetch_data).start()

def start_processing():
    global is_running, is_paused
    if not is_running:
        is_running = True
        is_paused = False
        threading.Thread(target=process_paragraphs).start()
    elif is_paused:
        is_paused = False

def pause_processing():
    global is_paused
    is_paused = True
    send_audio_control_api(PAUSE_API_URL)  # Call Pause API

def stop_processing():
    global is_running, is_paused, current_index
    is_running = False
    is_paused = False
    current_index = 0
    send_audio_control_api(STOP_API_URL)  # Call Stop API
    print("Processing stopped.")

# Tkinter GUI
def create_gui():
    global start_button, pause_button, stop_button, speed_scale

    root = Tk()
    root.title("Dass Ebook Reader")

    Label(root, text="Control Panel").pack(pady=10)

    Button(root, text="Launch Browser", command=open_browser, bg="blue", fg="white").pack(pady=5)
    Button(root, text="Fetch Data", command=fetch_data_button_action, bg="orange", fg="black").pack(pady=5)

    Label(root, text="Playback Speed").pack(pady=5)
    speed_scale = Scale(root, from_=0.5, to=3.0, resolution=0.1, orient=HORIZONTAL)
    speed_scale.set(1.6)  # Default playback speed
    speed_scale.pack(pady=5)

    start_button = Button(root, text="Start", command=start_processing, bg="green", fg="white", state=DISABLED)
    start_button.pack(pady=5)

    pause_button = Button(root, text="Pause", command=pause_processing, bg="yellow", fg="black", state=DISABLED)
    pause_button.pack(pady=5)

    stop_button = Button(root, text="Stop", command=stop_processing, bg="red", fg="white", state=DISABLED)
    stop_button.pack(pady=5)

    root.geometry("250x400")
    root.mainloop()

# Main
if __name__ == "__main__":
    # Step 1: Start GUI
    create_gui()
