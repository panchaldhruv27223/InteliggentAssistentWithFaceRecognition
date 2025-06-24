# import numpy as np
import speech_recognition as sr
import requests
import whisper
import say
import time

flag = 0

def is_connected():
    try:
        response = requests.get("http://www.google.com", timeout=5)
        if response.status_code == 200:
            return True
        else:
            return False

    except Exception as e:
        print(e)
        return False

# print("first function ",is_connected())

def capture_voice_input():
    print("Checking for internet")

    # start = time.time()
    flag = is_connected()

    # print("Internet checked at ", (time.time()-start))
    recognizer = sr.Recognizer()

    # print("Avaliabel microphones : ",sr.Microphone.list_microphone_names())

    with sr.Microphone() as source :
        
        print("Listening...")

        audio = recognizer.listen(source,phrase_time_limit=4)

        # print("not able to capture voice data error occure")

        # print("input taken from user")

        # print("Debug point 1")

    try:
        if flag:
            text = recognizer.recognize_google(audio,language="english")
        else:
            print("No internet connection.")
            text = recognizer.recognize_whisper(audio, model="base")

    except sr.UnknownValueError:
        # say.SpeakText("Sorry, I didn't understand that.")
        text = ""
    # print("Sorry, I didn't understand that.")
    
    except sr.RequestError as e:
        text = ""
        # print("Error; {0}".format(e))
    
    except Exception:
        text = ""

    return text


# while True :
#     print(capture_voice_input())