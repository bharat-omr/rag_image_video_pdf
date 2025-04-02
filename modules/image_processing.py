import google.generativeai as genai
import PIL.Image
from dotenv import load_dotenv
import os 
# API Key
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
#Gemini API
genai.configure(api_key=API_KEY)
def describe_image(image):
    try:
        model = genai.GenerativeModel("gemini-2.0-flash")
        prompt = "Describe this video in detail. What are the purpose and what his/her speak?"
        response = model.generate_content([prompt, image])
        return response.text.strip() if response.text else "No description generated"
    except Exception as e:
        return f"❌ Error: {str(e)}"
