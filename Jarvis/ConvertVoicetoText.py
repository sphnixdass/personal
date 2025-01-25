import speech_recognition as sr

def speech_to_text():
    # Create a Recognizer object
    r = sr.Recognizer()

    # Use the microphone as the audio source
    with sr.Microphone() as source:
        print("Please say something:")
        # Listen for audio from the microphone
        audio = r.listen(source)

        try:
            # Try to convert speech to text
            print("You said: " + r.recognize_google(audio))
        except sr.UnknownValueError:
            print("Sorry, I couldn't understand what you said.")
        except sr.RequestError as e:
            print("Error requesting results from Google Speech Recognition service; {0}".format(e))

speech_to_text()
