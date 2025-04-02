import os
import time
import whisper
import streamlit as st
import tempfile
import google.generativeai as genai
from dotenv import load_dotenv
from moviepy.editor import AudioFileClip
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter

# Load API Key
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise ValueError("API key not found. Set GOOGLE_API_KEY in the environment variables.")

genai.configure(api_key=API_KEY)

# Function to transcribe audio from video
def transcribe_audio(audio_path, model_size="small"):
    model = whisper.load_model(model_size)
    result = model.transcribe(audio_path)
    return result["text"]

# Function to extract audio from a video
def extract_audio_from_video(video_path):
    temp_audio_path = os.path.join(tempfile.gettempdir(), "extracted_audio.wav")
    video = AudioFileClip(video_path)
    video.write_audiofile(temp_audio_path)
    return temp_audio_path

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
        prompt = "Describe this video in detail. What is happening in it?"
        response = model.generate_content([prompt, video_file])
        
        genai.delete_file(video_file.name)  # Clean up
        return response.text if response.text else "No description generated."
    except Exception as e:
        st.error(f"❌ Error processing video: {str(e)}")
        return None

# Function to create FAISS vector store
def get_vectorstore(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    return FAISS.from_texts(texts=text_chunks, embedding=embeddings)

# Function to split text into chunks
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len
    )
    return text_splitter.split_text(text)

# Streamlit UI
def main():
    st.set_page_config(page_title="Video Processing App", page_icon="🎥")
    st.header("Video Processing: Transcription or Description 🎥")
    
    video_file = st.file_uploader("Upload a Video", type=["mp4", "avi", "mov"])
    
    option = st.radio("Select an Option:", ["Transcribe Audio", "Generate Video Description"])
    
    if video_file and st.button("Process Video"):
        with st.spinner("Processing... Please wait."):
            temp_video_path = os.path.join(tempfile.gettempdir(), video_file.name)
            with open(temp_video_path, "wb") as f:
                f.write(video_file.read())
            
            all_text = ""
            
            if option == "Transcribe Audio":
                audio_path = extract_audio_from_video(temp_video_path)
                transcription = transcribe_audio(audio_path)
                st.subheader("Transcription:")
                st.write(transcription)
                all_text += transcription
            
            elif option == "Generate Video Description":
                description = describe_video(temp_video_path)
                st.subheader("Video Description:")
                st.write(description)
                all_text += description if description else ""
            
            # Convert extracted content into vector embeddings
            text_chunks = get_text_chunks(all_text)
            vectorstore = get_vectorstore(text_chunks)
            st.success("✅ Embeddings created successfully!")
            
if __name__ == '__main__':
    main()
