import os
import google.generativeai as genai
from dotenv import load_dotenv
from libs.logger import Logger


class GeminiVision:
    def __init__(self) -> None:
        self.logger = Logger.get_logger('gemini_vision.log')
        self.model = None

    def configure_genai(self):
        # load the key from the .env file
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            self.logger.error("No API key found in the .env file")
            raise ValueError("No API key found in the .env file")
        genai.configure(api_key=api_key)

    def setup_model(self,temperature=0.1,top_p=1,top_k=32,max_output_tokens=4096):
        try:
            # Set up the model
            generation_config = {
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "max_output_tokens": max_output_tokens,
            }

            self.model = genai.GenerativeModel(model_name="gemini-pro-vision",generation_config=generation_config)
        except Exception as e:
            self.logger.error(f"Error setting up model: {e}")
            raise

    def generate_content(self,contents):
        self.logger.info(f"Generating contents")
        
        # Check model and contents for errors.
        if self.model is None:
            self.logger.error("Model is not initialized")
            raise ValueError("Model is not initialized")

        if contents is None:
            self.logger.error("Contents is not initialized")
            raise ValueError("Contents is not initialized")
        
        return self.model.generate_content(contents=contents)
