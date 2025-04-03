Agentic AI Integration for LMS

part -1 

"This Flask API allows you to store your course-specific data in a vector database and query it using AI. You can upload PDFs, images, and videos, extract meaningful information, and ask questions based on the stored data. This approach enables an AI-driven agent that provides contextual responses using a conversational retrieval system."


🚀 Overview

This project provides a Flask-based API that processes PDFs, images, and videos, extracts meaningful data, and enables conversational retrieval using FAISS vector databases and Google Gemini AI. The API supports:

PDF Processing: Extract text and store it in a vector database.

Image Processing: Generate descriptions using Gemini AI.

Video Processing:

Extract audio transcription using Whisper AI.

Generate video descriptions using Gemini AI.

Conversational Retrieval: Ask questions based on stored documents.

📂 Project Structure

│── /app

│   ├── __init__.py          # Initializes Flask app

│   ├── config.py            # API Key Configuration


│   ├── routes.py            # Defines Flask API routes

│   ├── utils.py             # Helper functions (image, audio, video processing)

│   ├── vectorstore.py       # FAISS vector database management

│── main.py                  # Entry point

│── requirements.txt         # Dependencies

│── README.md                # Documentation

Run the Flask Server

python main.py

📌 API Endpoints

🔹 Upload PDF

Endpoint: POST /upload_pdf

Description: Extracts text from PDFs and stores it in FAISS.

Request:

curl -X POST -F "files=@sample.pdf" http://localhost:8000/upload_pdf

Upload Image

Endpoint: POST /upload_image

Description: Generates an AI-powered description of the uploaded image.

Request:

curl -X POST -F "file=@image.jpg" http://localhost:8000/upload_image



{
  "description": "A beautiful sunset over the mountains.",
  "message": "Image description stored in vector database."
}

🔹 Upload Video

Endpoint: POST /upload_video

Description: Extracts audio transcription or video description.

Request:

curl -X POST -F "file=@video.mp4" -F "option=Transcription" http://localhost:8000/upload_video



{
  "result": "This video is about AI and machine learning...",
  "message": "Video description stored in vector database."
}

🔹 Ask a Question

Endpoint: POST /ask_question

Description: Queries stored data and retrieves relevant responses.

Request:

curl -X POST -H "Content-Type: application/json" -d '{"question": "What is the content of the PDF?"}' http://localhost:8000/ask_question

 Technologies Used

Flask (Backend API)

Google Gemini AI (Image & Video Descriptions, Conversational AI)

FAISS (Vector Database for Text Search)

Whisper AI (Audio Transcription)

PyPDF2 (PDF Text Extraction)

MoviePy (Audio Extraction from Video)



![Screenshot 2025-04-02 162110](https://github.com/user-attachments/assets/f82d9086-dceb-49e8-aa95-07bf1e836765)

![Screenshot 2025-04-02 162125](https://github.com/user-attachments/assets/cf786026-9229-4614-9c66-cde80485410c)

