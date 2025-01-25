from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from gtts import gTTS
import os
from pydub.playback import play
from threading import Thread
import time
import pygame
from pydub import AudioSegment
import tempfile

# Initialize pygame mixer
pygame.mixer.init()

app = FastAPI()

# Global variables for MP3 playback
current_mp3_file = None
play_thread = None
is_playing = False
should_stop = False


# Input model for text-to-MP3 conversion
class TextToMP3Request(BaseModel):
    text: str
    lang: str = "en"  # Default language is English


# Input model for setting playback speed
class PlaybackSpeedRequest(BaseModel):
    speed: float = 1.0  # Default speed (normal)


# Text-to-MP3 conversion API
@app.post("/convert-text-to-mp3/")
def convert_text_to_mp3(request: TextToMP3Request):
    try:
        global current_mp3_file
        tts = gTTS(text=request.text, lang=request.lang)
        current_mp3_file = "output.mp3"
        tts.save(current_mp3_file)
        time.sleep(1)
        return {"status": "success", "message": f"MP3 created: {current_mp3_file}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def adjust_speed(audio_file, speed=1.0):
    """Adjust playback speed of the audio file."""
    segment = AudioSegment.from_file(audio_file)
    segment = segment.speedup(playback_speed=speed)
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    segment.export(temp_file.name, format="mp3")
    return temp_file.name


# Play MP3 API
@app.post("/play-mp3/")
def play_mp3(request: PlaybackSpeedRequest):
    global current_mp3_file, play_thread, is_playing, should_stop

    if not current_mp3_file or not os.path.exists(current_mp3_file):
        raise HTTPException(status_code=400, detail="No MP3 file available to play")

    if is_playing:
        raise HTTPException(status_code=400, detail="An MP3 is already playing")

    def playback_worker():
        global is_playing, should_stop

        # Adjust speed and prepare audio file
        adjusted_file = adjust_speed(current_mp3_file, speed=request.speed)
        
        # Load and play the audio
        pygame.mixer.music.load(adjusted_file)
        pygame.mixer.music.set_volume(1.0)  # Set volume (0.0 to 1.0)
        pygame.mixer.music.play()
        
        is_playing = True
        should_stop = False
        
        # Check for stopping condition
        while pygame.mixer.music.get_busy():
            if should_stop:
                pygame.mixer.music.stop()
                break

        is_playing = False


        # from pydub import AudioSegment
        # from pydub.playback import play
        # segment = AudioSegment.from_file(current_mp3_file)
        # segment = segment.speedup(playback_speed=request.speed)

        # is_playing = True
        # should_stop = False
        # for chunk in segment[::1000]:  # Play in chunks to allow for stopping
        #     if should_stop:
        #         break
        #     play(chunk)
        #     # time.sleep(0.01)
        # is_playing = False

    play_thread = Thread(target=playback_worker)
    play_thread.start()
    return {"status": "success", "message": "Playing MP3"}


# Pause API
@app.post("/pause-mp3/")
def pause_mp3():
    global is_playing, should_stop

    if not is_playing:
        raise HTTPException(status_code=400, detail="No MP3 is currently playing")

    should_stop = True
    is_playing = False
    return {"status": "success", "message": "MP3 playback paused"}


# Stop API
@app.post("/stop-mp3/")
def stop_mp3():
    global should_stop, is_playing

    if not is_playing:
        raise HTTPException(status_code=400, detail="No MP3 is currently playing")

    should_stop = True
    is_playing = False
    return {"status": "success", "message": "MP3 playback stopped"}


# Check if MP3 is playing API
@app.get("/is-playing-mp3/")
def is_playing_mp3():
    global is_playing
    return {"is_playing": is_playing}
