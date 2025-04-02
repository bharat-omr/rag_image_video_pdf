from langchain_google_genai import ChatGoogleGenerativeAI

def get_chat_model():
    return ChatGoogleGenerativeAI(model="gemini-2.0-flash")
