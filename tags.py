from fastapi import FastAPI, File, UploadFile
import subprocess
import io
import csv
import re
import speech_recognition as sr
from pydantic import BaseModel
from pydub import AudioSegment
import matplotlib.pyplot as plt
import numpy as np
from translate import Translator

app = FastAPI()

class InputData(BaseModel):
    input: str

class VoiceInput(BaseModel):
    audio_file: UploadFile

csv_file_path = "/home/lenovo/sound-to-text/etperify_new - Sheet1.csv"




def load_csv_content(csv_file_path):
    with open(csv_file_path, 'r') as file:
        csv_reader = csv.DictReader(file)
        return list(csv_reader)

def extract_tags_from_csv(input_text, csv_content):
    generated_tags = set()

    for row in csv_content:
        csv_input = row.get("input")
        tags = row.get("tags")

        print(f"csv_input: {csv_input}, input_text: {input_text}")

        if csv_input and tags:
            # Check if any word from input_text is similar to any word in csv_input
            for word in input_text:
                for csv_word in csv_input.lower().split():
                    if word.lower() == csv_word:
                        tags_list = [tag.strip() for tag in tags.split(',')]
                        generated_tags.update(tags_list)

    return list(generated_tags)



def voice_to_tags(audio_file_path, csv_content):
    recognizer = sr.Recognizer()

    try:
        with sr.AudioFile(audio_file_path) as source:
            recognizer.adjust_for_ambient_noise(source, duration=1)
            audio_text = recognizer.record(source, duration=8)

        # Recognize audio using Google Web Speech API
        text = recognizer.recognize_google(audio_text, language="en-US", show_all=False)

        # Filter out non-alphabetic characters and extract words
        filtered_text = re.sub(r'[^a-zA-Z\s]', '', text)
        words = filtered_text.lower().split()

        # Translate the detected words
        translator = Translator(to_lang="en")
        translated_words = [translator.translate(word) for word in words]

        # Extract tags for words similar to the input from CSV content
        generated_tags = extract_tags_from_csv(translated_words, csv_content)

        return {"text": filtered_text, "translated_words": translated_words, "generated_tags": generated_tags}

    except sr.UnknownValueError:
        return {"text": "Speech Recognition could not understand audio", "generated_tags": []}
    except sr.RequestError as e:
        return {"text": f"Could not request results from Google Web Speech API; {e}", "generated_tags": []}
    except FileNotFoundError:
        return {"text": f"Error: File not found at {audio_file_path}", "generated_tags": []}


def text_input(text_or_audio, csv_content):
    if isinstance(text_or_audio, str):
        # If the input is text, directly use it
        text = text_or_audio.lower()
    else:
        # If the input is audio, perform speech recognition
        audio_data = text_or_audio.file.read()

        # Convert to PCM WAV using pydub for MP3 files
        audio = AudioSegment.from_mp3(io.BytesIO(audio_data))
        audio.export("audio_files/temp_audio.wav", format="wav")

        # Use speech recognition to convert audio to text
        recognizer = sr.Recognizer()
        with sr.AudioFile("audio_files/temp_audio.wav") as source:
            audio_text = recognizer.record(source)
        text = recognizer.recognize_google(audio_text)

    # Extract tags from CSV content
    generated_tags = extract_tags_from_csv(text, csv_content)

    return {"input": text, "generated_tags": generated_tags}


@app.post("/process_voice")
async def process_voice(audio_file: UploadFile = File(...)):
    # Save the uploaded audio file
    audio_data = await audio_file.read()

    # Convert to PCM WAV using ffmpeg
    converted_audio = subprocess.run(
        ["ffmpeg", "-i", "-", "-f", "wav", "-"],
        input=audio_data,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Convert bytes to audio file
    audio_file_path = "audio_files/temp_audio.wav"
    with open(audio_file_path, "wb") as sound_file:
        sound_file.write(converted_audio.stdout)

    # Load CSV content
    csv_content = load_csv_content(csv_file_path)

    # Extract generated tags from voice content
    result = voice_to_tags(audio_file_path, csv_content)

    return result

@app.post("/process_input")
async def process_input(input_data: InputData):
    # Load CSV content
    csv_content = load_csv_content(csv_file_path) 

    # Extract generated tags from input (text or audio)
    result = text_input(input_data.input, csv_content)

    return result
