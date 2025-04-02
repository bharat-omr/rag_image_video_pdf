import os
from flask import request, jsonify
from .utils import get_pdf_text, extract_audio_from_video, transcribe_audio, describe_image, describe_video
from .vectorstore import update_vectorstore, get_vectorstore
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_google_genai import ChatGoogleGenerativeAI
import PIL.Image

def register_routes(app):

    @app.route("/upload_pdf", methods=["POST"])
    def upload_pdf():
        files = request.files.getlist("files")

        if not files:
            return jsonify({"error": "No files uploaded"}), 400

        texts = [get_pdf_text([file]) for file in files]
        text_chunks = " ".join(texts).split("\n")
        
        update_vectorstore(text_chunks)
        return jsonify({"message": "PDF text processed and stored in vector database."})

    @app.route("/upload_video", methods=["POST"])
    def upload_video():
        file = request.files.get("file")
        option = request.form.get("option")

        if not file or not option:
            return jsonify({"error": "Missing file or processing option"}), 400

        temp_path = f"temp_{file.filename}"
        file.save(temp_path)

        if option == "Transcription":
            audio_path = extract_audio_from_video(temp_path)
            result = transcribe_audio(audio_path)
        elif option == "Description":
            result = describe_video(temp_path, app.config["GEMINI_MODEL"])
        else:
            result = "Invalid option"

        os.remove(temp_path)
        update_vectorstore([result])

        return jsonify({"result": result, "message": "Video description stored in vector database."})

    @app.route("/upload_image", methods=["POST"])
    def upload_image():
        file = request.files.get("file")

        if not file:
            return jsonify({"error": "No image file uploaded"}), 400

        image = PIL.Image.open(file).convert("RGB")
        description = describe_image(image, app.config["GEMINI_MODEL"])
        
        update_vectorstore([description])
        return jsonify({"description": description, "message": "Image description stored in vector database."})

    @app.route('/ask_question', methods=['POST'])
    def ask_question():
        data = request.get_json()
        question = data.get("question")

        if not question:
            return jsonify({"error": "Question is required"}), 400

        vectorstore = get_vectorstore()
        if vectorstore is None:
            return jsonify({"error": "No vectorstore found, upload files first."}), 400

        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=ChatGoogleGenerativeAI(model='gemini-2.0-flash'),
            retriever=vectorstore.as_retriever(),
            memory=ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        )

        response = conversation_chain({"question": question})
        chat_history = [msg.content for msg in response['chat_history']]

        return jsonify({"answer": chat_history})
