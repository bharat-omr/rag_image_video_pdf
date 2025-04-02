import time
import tempfile
import os
import google.generativeai as genai
from moviepy.editor import AudioFileClip
import whisper
from dotenv import load_dotenv
import streamlit as st
# API Key
load_dotenv()

API_KEY = os.getenv("GOOGLE_API_KEY")
#Gemini API
genai.configure(api_key=API_KEY)
# Function to extract audio from a video
def extract_audio_from_video(video_path, audio_path="extracted_audio.wav"):
    video = AudioFileClip(video_path)
    video.write_audiofile(audio_path)
    return audio_path

# Function to transcribe audio using Whisper
def transcribe_audio(audio_path, model_size="small"):
    model = whisper.load_model(model_size)
    result = model.transcribe(audio_path)
    return result["text"]


# Function to generate video description using Gemini
def describe_video(video_path):
    try:
        video_file = genai.upload_file(path=video_path, display_name="Uploaded Video")
        st.info("Processing video... This might take some time.")
        time.sleep(10)  # Allow some time for processing
        
        while video_file.state.name == "PROCESSING":
            time.sleep(10)
            video_file = genai.get_file(video_file.name)
        
        if video_file.state.name != "ACTIVE":
            st.error("Error: Video processing failed or is not active.")
            return None
        
        model = genai.GenerativeModel("gemini-2.0-flash")
        prompt = "Describe this video in detail. What are the purpose and what his/her speak?"
        response = model.generate_content([prompt, video_file])
        
        genai.delete_file(video_file.name)  # Clean up
        return response.text if response.text else "No description generated."
    except Exception as e:
        st.error(f"❌ Error processing video: {str(e)}")
        return None


# Process video: User chooses between transcription or description
def process_video(video_file, option):
    if video_file is not None:
        temp_dir = tempfile.gettempdir()
        temp_video_path = os.path.join(temp_dir, video_file.name)
        
        with open(temp_video_path, "wb") as f:
            f.write(video_file.read())
        
        if option == "Transcription":
            audio_path = extract_audio_from_video(temp_video_path)
            return transcribe_audio(audio_path)
        elif option == "Description":
            return describe_video(temp_video_path)
    return None
