import os
import whisper
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import tempfile
import time
import google.generativeai as genai
import PIL.Image
from moviepy.editor import AudioFileClip

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# API Key
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")

if not API_KEY:
    raise ValueError("API key not found. Set GOOGLE_API_KEY in the environment variables.")

#Gemini API
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("gemini-2.0-flash")

# Function to generate image descriptions
def describe_image(image):
    try:
        prompt = "Describe the content of the image in detail. Provide a meaningful and accurate description."
        response = model.generate_content([prompt, image])
        return response.text.strip() if response.text else "No description generated"
    except Exception as e:
        return f"❌ Error: {str(e)}"

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

# Extract text from PDFs
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Split text into chunks
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return text_splitter.split_text(text)

# Function to create FAISS vector store
def get_vectorstore(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    return FAISS.from_texts(texts=text_chunks, embedding=embeddings)

# User input handling
def handle_userinput(question):
    response = st.session_state.conversation({"question": question})
    st.session_state.chat_history = response['chat_history']
    st.write(response)

# Function to create a conversation chain
def get_conversation_chain(vectorstore):
    llm = ChatGoogleGenerativeAI(model='gemini-2.0-flash')
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )

# Main Streamlit App
def main():
    st.set_page_config(page_title="Chat with PDFs, Videos & Images", page_icon="📄🎥🖼️")
    st.header("Chat with PDFs, Videos & Images 📄🎥🖼️")

    user_question = st.text_input("Ask a question about your documents, videos, or images:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Upload your files")
        
        pdf_docs = st.file_uploader("Upload PDFs", accept_multiple_files=True, type=["pdf"])
        video_file = st.file_uploader("Upload a Video", type=["mp4", "avi", "mov"])
        image_files = st.file_uploader("Upload Images", accept_multiple_files=True, type=["png", "jpg", "jpeg"])

        video_processing_option = None
        if video_file:
            video_processing_option = st.radio("Select video processing option:", ("Transcription", "Description"))

        if st.button("Process"):
            with st.spinner("Processing..."):
                all_text = ""

                # Process PDFs
                if pdf_docs:
                    pdf_text = get_pdf_text(pdf_docs)
                    all_text += pdf_text
                
                # Process Video
                if video_file and video_processing_option:
                    video_output = process_video(video_file, video_processing_option)
                    all_text += video_output
                
                # Process Images
                if image_files:
                    st.subheader("Image Descriptions")
                    for image_file in image_files:
                        image = PIL.Image.open(image_file).convert("RGB")
                        description = describe_image(image)
                        st.image(image, caption="Uploaded Image", use_container_width=True)
                        all_text += "\n" + description

                # Convert all extracted text into vector embeddings
                text_chunks = get_text_chunks(all_text)
                vectorstore = get_vectorstore(text_chunks)

                # Store the conversation chain in session state
                st.session_state.conversation = get_conversation_chain(vectorstore)

if __name__ == '__main__':
    main()
