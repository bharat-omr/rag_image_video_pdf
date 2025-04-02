import os
import whisper
import time
import json
import google.generativeai as genai
import PIL.Image
from PyPDF2 import PdfReader
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from moviepy.editor import AudioFileClip
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
app = Flask(__name__)

#API Key
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")

if not API_KEY:
    raise ValueError("API key not found. Set GOOGLE_API_KEY in the environment variables.")

# Gemini model
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("gemini-2.0-flash")

#Store FAISS vector database 
vectorstore = None

# Embeddings
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

# Function to describe an image using Gemini
def describe_image(image):
    try:
        prompt = "Describe the content of the image in detail."
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

# Function to describe a video using Gemini
def describe_video(video_path):
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
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_text(text)

# Function to create/update FAISS vector store
def update_vectorstore(text_chunks):
    global vectorstore
    if vectorstore is None:
        vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    else:
        new_store = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
        vectorstore.merge_from(new_store)

# Function to get conversation chain
def get_conversation_chain(vectorstore):
    llm = ChatGoogleGenerativeAI(model='gemini-2.0-flash')
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    return ConversationalRetrievalChain.from_llm(llm=llm, retriever=vectorstore.as_retriever(), memory=memory)

# Flask API Endpoints

@app.route("/upload_pdf", methods=["POST"])
def upload_pdf():
    files = request.files.getlist("files")
    
    if not files:
        return jsonify({"error": "No files uploaded"}), 400

    texts = []
    for file in files:
        text = get_pdf_text([file])
        texts.append(text)

    combined_text = " ".join(texts)
    text_chunks = get_text_chunks(combined_text)

    update_vectorstore(text_chunks)
    
    return jsonify({"message": "PDF text processed and stored in vector database."})

@app.route("/upload_video", methods=["POST"])
def upload_video():
    file = request.files.get("file")
    option = request.form.get("option")  # "Transcription" or "Description"

    if not file or not option:
        return jsonify({"error": "Missing file or processing option"}), 400

    temp_path = f"temp_{file.filename}"
    file.save(temp_path)
    
    if option == "Transcription":
        audio_path = extract_audio_from_video(temp_path)
        result = transcribe_audio(audio_path)
    elif option == "Description":
        result = describe_video(temp_path)
    else:
        result = "Invalid option"

    os.remove(temp_path)

    # Store description in FAISS
    update_vectorstore([result])

    return jsonify({"result": result, "message": "Video description stored in vector database."})

@app.route("/upload_image", methods=["POST"])
def upload_image():
    file = request.files.get("file")

    if not file:
        return jsonify({"error": "No image file uploaded"}), 400

    image = PIL.Image.open(file).convert("RGB")
    description = describe_image(image)

    # Store description in FAISS
    update_vectorstore([description])
    
    return jsonify({"description": description, "message": "Image description stored in vector database."})

@app.route('/ask_question', methods=['POST'])
def ask_question():
    data = request.get_json()
    question = data.get("question")

    if not question:
        return jsonify({"error": "Question is required"}), 400

    conversation_chain = get_conversation_chain(vectorstore)
    response = conversation_chain({"question": question})
    
    # Convert chat history to a serializable format
    chat_history = [msg.content for msg in response['chat_history']]

    return jsonify({"answer": chat_history})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8000, debug=True)
