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


![Screenshot 2025-04-02 162110](https://github.com/user-attachments/assets/f82d9086-dceb-49e8-aa95-07bf1e836765)

![Screenshot 2025-04-02 162125](https://github.com/user-attachments/assets/cf786026-9229-4614-9c66-cde80485410c)

