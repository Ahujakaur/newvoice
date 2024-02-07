from fastapi import FastAPI, File, UploadFile
import subprocess
import io
import csv
from pydantic.typing import Schema
import os
import re
import speech_recognition as sr
from pydantic import BaseModel
from pydub import AudioSegment
from typing import Union
from nltk.tokenize import word_tokenize  # Add this import statement

import matplotlib.pyplot as plt
import numpy as np
from translate import Translator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import MultiLabelBinarizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model_name = 'bert-base-nli-mean-tokens'
bert_model = SentenceTransformer(model_name)
app = FastAPI()
csv_content = None
class InputData(BaseModel):
    input: str
    
class VoiceInput(BaseModel):
    audio_file: UploadFile

common_words_to_ignore = set(["help", "need", "i","develope"])

def load_csv_content(csv_file_path):
    with open(csv_file_path, 'r') as file:
        csv_reader = csv.DictReader(file)
        return list(csv_reader)

def clean_tags(tags):
    # Remove +ACI- characters from tags
    cleaned_tags = [tag.replace("+ACI-", "").strip() for tag in tags]
    # Remove non-printable characters using regular expression
    cleaned_tags = [re.sub(r'[\x00-\x1F\x7F-\x9F]', '', tag) for tag in cleaned_tags]
    return cleaned_tags

csv_file_path = "expertify_new-Sheet1.csv"
csv_content = load_csv_content(csv_file_path)
input_texts = [row["input"] for row in csv_content]
tags = [row["tags"].split(',') for row in csv_content]
mlb = MultiLabelBinarizer()
binary_tags = mlb.fit_transform(tags)

vectorizer = TfidfVectorizer()
classifier = OneVsRestClassifier(SVC(kernel='linear'))
model = make_pipeline(vectorizer, classifier)
model.fit(input_texts, binary_tags)


def extract_tags_from_csv_bert(input_text, csv_content):
    # Embed the input text using BERT
    input_embedding = bert_model.encode([input_text])[0]

    # Calculate the similarity between input text embedding and CSV text embeddings
    similarities = []
    for row in csv_content:
        csv_input = row.get("input")
        csv_embedding = bert_model.encode([csv_input])[0]
        similarity = cosine_similarity([input_embedding], [csv_embedding])[0][0]
        similarities.append((csv_input, similarity))

    # Sort the similarities in descending order
    similarities.sort(key=lambda x: x[1], reverse=True)

    # Calculate the total similarity score for normalization
    total_similarity = sum(similarity for _, similarity in similarities)

    # Extract top similar tags with their percentage of similarity
    top_similar_tags = []
    for csv_input, similarity in similarities[:7]:
        similarity_percent = (similarity / total_similarity) * 100
        cleaned_tag = clean_tags([csv_input])[0]  # Clean the extracted tag
        top_similar_tags.append((cleaned_tag, similarity_percent))

    return top_similar_tags

# Define the rest of your functions and endpoints here

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

        # Extract tags for words similar to the input from CSV content
        generated_tags = extract_tags_from_csv_bert(filtered_text, csv_content)

        return {"text": filtered_text, "generated_tags": generated_tags[:10]}

    except sr.UnknownValueError:
        return {"text": "Speech Recognition could not understand audio", "generated_tags": []}
    except sr.RequestError as e:
        return {"text": f"Could not request results from Google Web Speech API; {e}", "generated_tags": []}
    except FileNotFoundError:
        return {"text": f"Error: File not found at {audio_file_path}", "generated_tags": []}

def text_input(input_data, csv_content):
    if isinstance(input_data, str):
        # Tokenize the input text and filter out common words
        input_text = [word.lower() for word in word_tokenize(input_data.strip().lower()) if word.lower() not in common_words_to_ignore]
    elif isinstance(input_data, UploadFile):
        # If the input is a file, read its content, tokenize it, and filter out common words
        input_text = [word.lower() for word in word_tokenize(input_data.file.read().decode("utf-8").strip().lower()) if word.lower() not in common_words_to_ignore]
    else:
        return {"error": "Invalid input type"}  # Return an error for invalid input type

    # Initialize an empty set to store the generated tags
    generated_tags = set()

    # Check if any word from input_text is similar to any word in CSV and extract corresponding tags
    for row in csv_content:
        csv_input = row.get("input")
        tags = row.get("tags")

        # Split the CSV input into individual words
        csv_words = csv_input.lower().split()

        # Check if any word from input_text is a substring of any word in csv_input
        for word in input_text:
            for csv_word in csv_words:
                if word in csv_word:
                    # If a match is found, add corresponding tags to the set
                    tags_list = [tag.strip() for tag in tags.split(',')]
                    generated_tags.update(tags_list)
                    break  # Exit the loop once a match is found for efficiency

    return {"input": input_text, "generated_tags": list(generated_tags)}

@app.post("/process_voice")
async def process_voice(audio_file: UploadFile = File(...)):
    csv_file_path = "etperify_new - Sheet1.csv"  # Set the CSV file path here
    
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

@app.post("/predict")
async def predict(input_data: InputData):
    # Process the input data (text or audio)
    input_text = input_data.input
    
    # Extract tags for the input text from CSV content using BERT embeddings
    generated_tags = extract_tags_from_csv_bert(input_text, csv_content)
    
    # Make predictions using your trained model
    predictions = model.predict([input_text])  # Adjust this according to your model
    
    # Convert binary predictions back to tags
    predicted_tags = mlb.inverse_transform(predictions)
    
    # Retrieve only the most similar 10 tags
    if predicted_tags:
        predicted_tags = predicted_tags[0][:7]  # Take only the first 10 tags
    
    # Return the predictions
    return {"input": input_text, "generated_tags": generated_tags, "predicted_tags": predicted_tags}
