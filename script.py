import time
from libs.logger import Logger
from libs.gemini_vision import GeminiVision
from libs.image_cv2 import ImageCV2
from pathlib import Path
from PIL import Image
from io import BytesIO
import traceback
        
# Set up logging
logger = Logger.get_logger('gemini_vision.log')

def validate_image(image_path):
    if not image_path.exists():
        logger.error(f"Could not find image: {image_path}")
        raise FileNotFoundError(f"Could not find image: {image_path}")

import subprocess
import time
def check_file_type(file_path):
    try:
        result = subprocess.run(['file', file_path], capture_output=True, text=True)
        return result.stdout
    except Exception as e:
        print(f"An error occurred: {e}")


def main():
    logger.info("Starting Gemini Vision")
    gemini_vision = GeminiVision()

    # Configure GenAI
    gemini_vision.configure_genai()

    # Setup the model
    gemini_vision.setup_model(temperature=0.1,top_p=1,top_k=32,max_output_tokens=4096)

    # Capture the image from the webcam
    web_image = None
    web_cam = ImageCV2()
    web_image_file = "web_image.png"
    web_image = web_cam.capture_image_from_webcam(web_image_file)
    if web_image is None:
        raise ValueError("Could not capture image from webcam")
    
    # convert web_image from RGB to RGBA
    web_image = web_image.convert("RGBA")
    
    # Validate that an image is present
    image_path = Path(web_image_file)
    validate_image(image_path)
    
    # Open the image
    logger.info(f"Trying to open image: {web_image_file}")
    web_image = Image.open(web_image_file)
    
    logger.info(f"Initializing image prompt")
    image_prompt = "Describe this image and what is this image about and what you see? and be very precise and short."
    image_contents = [image_prompt,web_image]
    
    # Generate the content
    logger.info(f"Generating Image content")
    response = gemini_vision.generate_content(image_contents)
    if 'error' in response:
        raise ValueError(f"An error occurred: {response}")
    else:
        logger.info(f"Gemini:\n{response.text}")
            

# Run the main function repeatedly after an interval of 15 seconds
if __name__ == "__main__":
    gemini_request_count = 0
    
    while True:
        try:
            main()
            gemini_request_count += 1
            logger.info(f"Gemini requests: {gemini_request_count}")
        except Exception as exception:
            traceback.print_exc()
            logger.error(f"An error occurred: {str(exception)}")
        
        # Wait for 15 seconds before running again
        time.sleep(15)
