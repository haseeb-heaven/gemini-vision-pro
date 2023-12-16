import os
from pathlib import Path
import google.generativeai as genai
from dotenv import load_dotenv
from libs.logger import setup_logger
from PIL import Image
from io import BytesIO
    
# Setup the logger
logger = setup_logger("gemini_vision.log")

def configure_genai():
    # load the key from the .env file
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.error("No API key found in the .env file")
        raise ValueError("No API key found in the .env file")
    genai.configure(api_key=api_key)

def setup_model(generation_config):
    return genai.GenerativeModel(model_name="gemini-pro-vision",generation_config=generation_config)

def validate_image(image_path):
    if not image_path.exists():
        logger.error(f"Could not find image: {image_path}")
        raise FileNotFoundError(f"Could not find image: {image_path}")

def generate_content(model,contents):
    logger.info(f"Generating contents")

    return model.generate_content(
        contents=contents,
    )

def main():
    logger.info("Starting Gemini Vision")
    
    try:
        # Configure GenAI
        configure_genai()

        # Set up the model
        generation_config = {
            "temperature": 0.1,
            "top_p": 1,
            "top_k": 32,
            "max_output_tokens": 4096,
        }

        # Setup the model
        model = setup_model(generation_config)

        # Validate that an image is present
        image_name = "image.png"
        image_path = Path(image_name)
        validate_image(image_path)
        
        # Open the image
        logger.info(f"Trying to open image: {image_name}")
        image = Image.open(image_name)
        
        logger.info(f"Initializing image prompt")
        image_prompt = "Describe this image and what is this image about and what you see? and be very descriptive and specific."
        image_contents = [image_prompt,image]
        
        # Generate the content
        response = generate_content(model, image_contents)
        if 'error' in response:
            raise ValueError(f"An error occurred: {response}")
        else:
            logger.info(f"Gemini:\n{response.text}")
            
    except Exception as exception:
        logger.error(f"An error occurred: {str(exception)}")

# Run the main function
if __name__ == "__main__":
    try:
        main()
    except Exception as exception:
        import traceback
        logger.error(f"An error occurred: {str(exception)}")
