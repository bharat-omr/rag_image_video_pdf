import whisper
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from google.generativeai import configure, GenerativeModel

# Configure Gemini API (Replace with your API Key)
configure(api_key="")

def transcribe_video(video_path, model_size="small"):
    """Transcribes video audio using Whisper."""
    model = whisper.load_model(model_size)
    result = model.transcribe(video_path)
    return result["text"]

def embed_and_store(text_chunks, embedding_model="all-MiniLM-L6-v2"):
    """Embeds text chunks and stores them in FAISS."""
    model = SentenceTransformer(embedding_model)
    embeddings = model.encode(text_chunks)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    return index, text_chunks

def retrieve_relevant_chunks(query, index, text_chunks, embedding_model="all-MiniLM-L6-v2", top_k=3):
    """Retrieves relevant text chunks based on query."""
    model = SentenceTransformer(embedding_model)
    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding, top_k)
    return [text_chunks[i] for i in indices[0]]

def generate_response(query, retrieved_text, model_name="gemini-pro"):
    """Generates response using Gemini based on retrieved text."""
    model = GenerativeModel(model_name)
    prompt = f"Context: {retrieved_text}\n\nQuestion: {query}\nAnswer:"
    response = model.generate_content(prompt)
    return response.text

# Example Usage
video_path = "videoplayback.mp4"
transcript = transcribe_video(video_path)
text_chunks = transcript.split(". ")  # Splitting into sentences for simplicity
index, stored_texts = embed_and_store(text_chunks)
query = "What is the main topic of the video?"
retrieved_text = retrieve_relevant_chunks(query, index, stored_texts)
response = generate_response(query, retrieved_text)
print("AI Response:", response)
