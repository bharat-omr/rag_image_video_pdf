import whisper
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain, LLMChain
import pdfplumber
import os
from moviepy.editor import AudioFileClip
import tempfile
import PIL.Image

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")


def extract_audio_from_video(video_path, audio_path="extracted_audio.wav"):
    """Extracts audio from a video file."""
    video = AudioFileClip(video_path)
    video.write_audiofile(audio_path)
    return audio_path

def transcribe_audio(audio_path, model_size="small"):
    """Transcribes audio using OpenAI's Whisper model."""
    model = whisper.load_model(model_size)
    result = model.transcribe(audio_path)
    return result["text"]

def process_video(video_file):
    """Save uploaded video to a temporary file, extract audio, and transcribe."""
    if video_file is not None:
        # Define a temp directory
        temp_dir = tempfile.gettempdir()
        temp_video_path = os.path.join(temp_dir, video_file.name)

        # Save the uploaded video file
        with open(temp_video_path, "wb") as f:
            f.write(video_file.read())

        # Extract audio and transcribe
        audio_path = extract_audio_from_video(temp_video_path)
        transcription = transcribe_audio(audio_path)
        return transcription
    return None

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def handle_userinput(question):
    response = st.session_state.conversation({"question": question})
    st.session_state.chat_history = response['chat_history']
    st.write(response)

def get_conversation_chain(vectorstore):
    llm = ChatGoogleGenerativeAI(model='gemini-2.0-flash')
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with PDFs & Videos", page_icon=":books:")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with PDFs & Videos :books:🎥")

    user_question = st.text_input("Ask a question about your documents or videos:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Upload your documents or videos")
        pdf_docs = st.file_uploader("Upload PDFs", accept_multiple_files=True, type=["pdf"])
        video_file = st.file_uploader("Upload a Video", type=["mp4", "avi", "mov"])

        if st.button("Process"):
            with st.spinner("Processing"):
                all_text = ""
                if pdf_docs:
                    raw_text = get_pdf_text(pdf_docs)
                    all_text += raw_text
                if video_file:
                    transcription = process_video(video_file)
                    all_text += transcription
                
                text_chunks = get_text_chunks(all_text)
                st.write(text_chunks)
                vectorstore = get_vectorstore(text_chunks)
                st.session_state.conversation = get_conversation_chain(vectorstore)

if __name__ == '__main__':
    main()
