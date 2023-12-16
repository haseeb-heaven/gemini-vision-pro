import time
from libs.logger import Logger
from libs.gemini_vision import GeminiVision
from libs.image_cv2 import ImageCV2
from pathlib import Path
from PIL import Image
from io import BytesIO

# Set up logging
logger = Logger.get_logger('gemini_vision.log')

def validate_image(image_path):
    if not image_path.exists():
        logger.error(f"Could not find image: {image_path}")
        raise FileNotFoundError(f"Could not find image: {image_path}")
    
def main():
    logger.info("Starting Gemini Vision")
    gemini_vision = GeminiVision()
    
    try:
        # Configure GenAI
        gemini_vision.configure_genai()

        # Setup the model
        gemini_vision.setup_model(temperature=0.1,top_p=1,top_k=32,max_output_tokens=4096)

        # Capture the image from the webcam
        web_image = None
        # web_cam = ImageCV2()
        # web_image_file = "web_image.png"
        # web_image = web_cam.capture_image_from_webcam(web_image_file)
        # if web_image is None:
        #     raise ValueError("Could not capture image from webcam")
        
        # Use the default image if the webcam image is not available
        if web_image is None:
            # Validate that an image is present
            image_name = "image.png"
            image_path = Path(image_name)
            validate_image(image_path)
            
            # Open the image
            logger.info(f"Trying to open image: {image_name}")
            web_image = Image.open(image_name)
        
        logger.info(f"Initializing image prompt")
        image_prompt = "Describe this image and what is this image about and what you see? and be very descriptive and specific."
        image_contents = [image_prompt,web_image]
        
        # Generate the content
        response = gemini_vision.generate_content(image_contents)
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
