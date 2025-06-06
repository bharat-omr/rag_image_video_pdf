import streamlit as st
import tempfile
import os
from dotenv import load_dotenv
from modules.pdf_processing import get_pdf_text
from modules.video_processing import process_video
from modules.image_processing import describe_image
from modules.vectorstore import get_text_chunks, get_vectorstore
from modules.chat import get_conversation_chain
import PIL.Image
import time



os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
load_dotenv()

st.set_page_config(page_title="Agentic AI Integration for LMS", page_icon="📄🎥🖼️")
st.header("Agentic AI Integration for LMS 📄🎥🖼️")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

def handle_userinput(question):
    if "conversation" in st.session_state:
        response = st.session_state.conversation({"question": question})
        bot_response = response.get("answer", "I'm not sure, please try again.")

        # Append user message
        st.session_state.messages.append({"role": "user", "content": question})
        
        # Append AI response
        st.session_state.messages.append({"role": "assistant", "content": bot_response})

        # Stream output dynamically
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            for word in bot_response.split():
                full_response += word + " "
                message_placeholder.markdown(full_response)
                time.sleep(0.05)  # Simulating the streaming effect

def main():
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

                if pdf_docs:
                    pdf_text = get_pdf_text(pdf_docs)
                    all_text += pdf_text

                if video_file and video_processing_option:
                    st.video(video_file)
                    video_output = process_video(video_file, video_processing_option)
                    all_text += video_output

                if image_files:
                    st.subheader("Image Descriptions")
                    for image_file in image_files:
                        image = PIL.Image.open(image_file).convert("RGB")
                        description = describe_image(image)
                        st.image(image, caption="Uploaded Image", use_container_width=True)
                        all_text += "\n" + description

                text_chunks = get_text_chunks(all_text)
                vectorstore = get_vectorstore(text_chunks)
                st.session_state.conversation = get_conversation_chain(vectorstore)
                st.success("Processing complete! You can now ask questions.")

    # Chat Interface
    st.subheader("Chat with AI")
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Get user input
    user_question = st.chat_input("Ask a question...")
    if user_question:
        with st.chat_message("user"):
            st.write(user_question)
        
        handle_userinput(user_question)

if __name__ == "__main__":
    main()
