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

csv_file_path = "expertify_new-Sheet1.csv"




def load_csv_content(csv_file_path):
    with open(csv_file_path, 'r') as file:
        csv_reader = csv.DictReader(file)
        return list(csv_reader)

def extract_tags_from_csv(input_text, csv_content):
    generated_tags = set()

    for row in csv_content:
        csv_input = row.get("input")
        tags = row.get("tags")

        if csv_input and tags:
            # Check if any word from input_text is similar to any word in csv_input
            for word in input_text.split():
                if word.lower() in csv_input.lower():
                    # If a word from input_text is found in csv_input, add corresponding tags
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

        # Join the translated words into a single string
        translated_text = " ".join(translated_words)

        # Extract tags for words similar to the input from CSV content
        generated_tags = extract_tags_from_csv(translated_text, csv_content)

        return {"text": filtered_text, "translated_words": translated_words, "generated_tags": generated_tags}

    except sr.UnknownValueError:
        return {"text": "Speech Recognition could not understand audio", "generated_tags": []}
    except sr.RequestError as e:
        return {"text": f"Could not request results from Google Web Speech API; {e}", "generated_tags": []}
    except FileNotFoundError:
        return {"text": f"Error: File not found at {audio_file_path}", "generated_tags": []}



def text_input(input_data, csv_content):
    if isinstance(input_data, str):
        # If the input is a string, directly use it after stripping leading/trailing whitespace
        input_text = input_data.strip().lower()
    elif isinstance(input_data, UploadFile):
        # If the input is a file, read its content and strip leading/trailing whitespace
        input_text = input_data.file.read().decode("utf-8").strip().lower()
    else:
        return {"error": "Invalid input type"}  # Return an error for invalid input type

    print("Input text:", input_text)

    # Initialize an empty set to store the generated tags
    generated_tags = set()

    # Check if any word from input_text is similar to any word in CSV and extract corresponding tags
    for row in csv_content:
        csv_input = row.get("input")
        tags = row.get("tags")

        print("CSV input:", csv_input)

        # Split the CSV input into individual words
        csv_words = csv_input.lower().split()

        # Check if any word from input_text is a substring of any word in csv_input
        for word in input_text.split():
            print("Checking word:", word)
            for csv_word in csv_words:
                if word in csv_word:
                    # If a match is found, add corresponding tags to the set
                    tags_list = [tag.strip() for tag in tags.split(',')]
                    generated_tags.update(tags_list)
                    break  # Exit the loop once a match is found for efficiency

    print("Generated tags:", generated_tags)

    return {"input": input_text, "generated_tags": list(generated_tags)}




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

    # Extract generated tags from input (text or file)
    result = text_input(input_data.input, csv_content)

    return result