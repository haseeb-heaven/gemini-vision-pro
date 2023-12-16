import os
from pathlib import Path
import google.generativeai as genai
from dotenv import load_dotenv
from logger import setup_logger
from PIL import Image

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

def generate_content(model, prompt_parts):
    logger.info(f"Generating content with prompt: {prompt_parts}")
    return model.generate_content(prompt_parts)

def main():
    logger.info("Starting Gemini Vision")
    
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

    from io import BytesIO
    # Load the image file
    image = Image.open(image_path)

    # Convert the PngImageFile object to bytes
    image_bytes_io = BytesIO()
    image.save(image_bytes_io, format='PNG')
    image_bytes = image_bytes_io.getvalue()
    
    image_parts = [
        {
            "mime_type": "image/png",
            "data": image_bytes
        },
    ]

    prompt_parts = [
        image_parts[0],
        "\nCan you describe this image and what is this image about and what you see? and be very descriptive and specific.",
    ]

    try:
        response = generate_content(model, prompt_parts)
        if not response.ok:
            logger.error(f"An error occurred: {response.text}")
            raise ValueError(f"An error occurred: {response.text}")
        else:
            logger.info(f"Response: {response.text}")
    except Exception as exception:
        print(f"An error occurred: {str(exception)}")

# Run the main function
if __name__ == "__main__":
    try:
        main()
    except Exception as exception:
        import traceback
        logger.error(f"An error occurred: {str(exception)}")
