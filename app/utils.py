import whisper
import time
import PIL.Image
from PyPDF2 import PdfReader
from moviepy.editor import AudioFileClip
import google.generativeai as genai

# Describe an image using Gemini
def describe_image(image, model):
    try:
        prompt = "Describe the content of the image in detail."
        response = model.generate_content([prompt, image])
        return response.text.strip() if response.text else "No description generated"
    except Exception as e:
        return f"❌ Error: {str(e)}"

# Extract text from PDFs
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Extract audio from a video
def extract_audio_from_video(video_path, audio_path="extracted_audio.wav"):
    video = AudioFileClip(video_path)
    video.write_audiofile(audio_path)
    return audio_path

# Transcribe audio using Whisper
def transcribe_audio(audio_path, model_size="small"):
    model = whisper.load_model(model_size)
    result = model.transcribe(audio_path)
    return result["text"]

# Describe a video using Gemini
def describe_video(video_path, model):
    try:
        video_file = genai.upload_file(path=video_path, display_name="Uploaded Video")
        time.sleep(10) 
        
        while video_file.state.name == "PROCESSING":
            time.sleep(10)
            video_file = genai.get_file(video_file.name)
        
        if video_file.state.name != "ACTIVE":
            return "Error: Video processing failed."
        
        prompt = "Describe this video in detail."
        response = model.generate_content([prompt, video_file])
        
        genai.delete_file(video_file.name)  # Clean up
        return response.text if response.text else "No description generated."
    except Exception as e:
        return f"❌ Error processing video: {str(e)}"
