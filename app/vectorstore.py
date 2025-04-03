from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import google.generativeai as genai
import os

load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=API_KEY)
print("API Key:", API_KEY)
vectorstore = None
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

def update_vectorstore(text_chunks):
    global vectorstore
    if vectorstore is None:
        vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    else:
        new_store = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
        vectorstore.merge_from(new_store)

def get_vectorstore():
    return vectorstore
