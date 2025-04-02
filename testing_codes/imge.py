import os
import PIL.Image
from dotenv import load_dotenv
import google.generativeai as genai  # Replace with actual AI model library

# Load API Key from .env file
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")

if not API_KEY:
    raise ValueError("API key not found. Set GOOGLE_API_KEY in the environment variables.")

# Configure Google Gemini API
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("gemini-2.0-flash")

def describe_image(image_path):
    """Generate a description for a single image using an AI model."""
    try:
        image = PIL.Image.open(image_path).convert("RGB")
        prompt = "Describe the content of the image in detail. Provide a meaningful and accurate description."
        response = model.generate_content([prompt, image])
        description = response.text.strip() if response.text else "No description generated"
        
        return description
    except Exception as e:
        return {"image": image_path, "description": f"❌ Error: {str(e)}"}

def describe_images_in_folders(folder_paths):
    """Generate descriptions for all images in multiple folders."""
    all_descriptions = {}
    
    for folder in folder_paths:
        if not os.path.isdir(folder):
            all_descriptions[folder] = "❌ Error: Folder not found"
            continue

        image_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(('png', 'jpg', 'jpeg'))]
        folder_descriptions = {img: describe_image(img) for img in image_files}
        all_descriptions[folder] = folder_descriptions
    
    return all_descriptions

# Example usage
description = describe_images_in_folders(["extracted_data/image_folder"])
print(description)