import os
from dotenv import load_dotenv
import google.generativeai as genai

def initialize_api(app):
    load_dotenv()
    API_KEY = os.getenv("GOOGLE_API_KEY")

    if not API_KEY:
        raise ValueError("API key not found. Set GOOGLE_API_KEY in environment variables.")

    genai.configure(api_key=API_KEY)
    app.config["GEMINI_MODEL"] = genai.GenerativeModel("gemini-2.0-flash")
