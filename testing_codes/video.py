import google.generativeai as genai
import os
import time
import argparse # For handling command-line arguments
from dotenv import load_dotenv
# --- Configuration ---
# Get API Key from environment variable (recommended)
# Load API Key
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
MODEL_NAME = "gemini-2.0-flash" # Use a model that supports video input

# Default prompt if none is provided via command line
DEFAULT_PROMPT = "Describe this video in detail. What are the purpose and what his/her speak?"

def describe_video(video_file_path: str, api_key: str, prompt: str):
    """
    Uploads a video file and uses the Gemini model to generate a description.

    Args:
        video_file_path: Path to the local video file.
        api_key: Your Google AI Studio API Key.
        prompt: The instruction for the Gemini model.

    Returns:
        The generated description string, or None if an error occurs.
    """
    if not api_key:
        print("Error: GOOGLE_API_KEY environment variable not set.")
        print("Please set the environment variable or pass the API key directly.")
        # You could add logic here to take the key as input if needed
        return None

    try:
        # Configure the generative AI client
        genai.configure(api_key=api_key)

        print(f"Uploading file: {video_file_path}...")
        # Upload the video file to the Gemini API File Service
        # Note: Uploading can take time depending on file size and network speed
        # The API has limits on file size (check documentation for current limits)
        video_file = genai.upload_file(path=video_file_path, display_name="Sample Video")
        print(f"Completed upload: {video_file.uri}") # URI is internal reference

        # --- Wait for file processing (Important!) ---
        # Sometimes the file needs processing time after upload before it's ready.
        print("Waiting for file processing...")
        while video_file.state.name == "PROCESSING":
            time.sleep(10) # Check every 10 seconds
            # Get the latest status of the file
            try:
                # Re-fetch the file object to get updated state
                video_file = genai.get_file(video_file.name)
            except Exception as e:
                print(f"Error checking file status: {e}")
                print("Proceeding anyway, but generation might fail if file not ready.")
                break # Exit the loop if status check fails

        if video_file.state.name == "FAILED":
            print("Error: File processing failed.")
            # Clean up the failed file upload
            try:
                genai.delete_file(video_file.name)
                print(f"Deleted failed file resource: {video_file.name}")
            except Exception as del_e:
                print(f"Error deleting failed file resource {video_file.name}: {del_e}")
            return None
        elif video_file.state.name != "ACTIVE":
             print(f"Warning: File is not active (state: {video_file.state.name}). Generation might fail.")
        else:
            print("File is active and ready.")


        # --- Generate Content ---
        print("Generating description...")
        # Create the generative model instance
        model = genai.GenerativeModel(model_name=MODEL_NAME)

        # Make the request to the model, including the prompt and the uploaded file
        # The model understands the file reference and processes the video
        response = model.generate_content([prompt, video_file], request_options={"timeout": 600}) # Increase timeout for video

        # --- Clean up the uploaded file ---
        print(f"Deleting uploaded file resource: {video_file.name}...")
        try:
            genai.delete_file(video_file.name)
            print("File resource deleted successfully.")
        except Exception as del_e:
            print(f"Error deleting file resource {video_file.name}: {del_e}")


        # --- Process and Return Response ---
        if response.parts:
             return response.text
        else:
            print("Error: No content generated. The response was empty.")
            print("Full Response Details:", response) # Print details for debugging
            # You might want to check response.prompt_feedback here for blocked content etc.
            if hasattr(response, 'prompt_feedback') and response.prompt_feedback:
                print("Prompt Feedback:", response.prompt_feedback)
            return None

    except FileNotFoundError:
        print(f"Error: Video file not found at '{video_file_path}'")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        # If the file was uploaded before the error, try to delete it
        if 'video_file' in locals() and hasattr(video_file, 'name'):
             print(f"Attempting to clean up uploaded file resource: {video_file.name}...")
             try:
                 genai.delete_file(video_file.name)
                 print("File resource deleted during error cleanup.")
             except Exception as del_e:
                 print(f"Error deleting file resource {video_file.name} during cleanup: {del_e}")
        return None


# --- Main execution block ---
if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Describe a video using the Gemini API.")
    parser.add_argument("video_path", help="Path to the video file.")
    parser.add_argument("-p", "--prompt", default=DEFAULT_PROMPT, help="Prompt for the Gemini model.")
    parser.add_argument("-k", "--api_key", default=API_KEY, help="Google AI Studio API Key (optional, overrides environment variable).")

    # Parse arguments
    args = parser.parse_args()

    # Call the main function
    description = describe_video(args.video_path, args.api_key, args.prompt)

    # Print the result
    if description:
        print("\n--- Video Description ---")
        print(description)
    else:
        print("\nFailed to generate description.")