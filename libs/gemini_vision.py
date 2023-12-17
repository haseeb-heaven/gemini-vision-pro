import os
import google.generativeai as genai
from dotenv import load_dotenv
from libs.logger import Logger


class GeminiVision:
    def __init__(self,api_key=None,temperature=0.1,top_p=1,top_k=32,max_output_tokens=4096) -> None:
        self.logger = Logger.get_logger('gemini_vision_pro.log')
        self.logger.info(f"Initializing Gemini Vision")
        self.model = None
        self.api_key = api_key
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.max_output_tokens = max_output_tokens
        
        self.logger.info(f"temperature: {self.temperature}")
        self.logger.info(f"top_p: {self.top_p}")
        self.logger.info(f"top_k: {self.top_k}")
        self.logger.info(f"max_output_tokens: {self.max_output_tokens}")
        
        if self.api_key is None:
            self.logger.error("API key is not initialized")

            # load the key from the .env file
            load_dotenv()
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                self.logger.error("No API key found in the .env file")
                raise ValueError("No API key found in the .env file")
        
        self.logger.info(f"Gemini Vision configured success")
        genai.configure(api_key=api_key)
        
        self.logger.info(f"Setting up model")
        self.setup_model()
        self.logger.info(f"Model setup success")

    def setup_model(self):
        try:
            # Set up the model
            generation_config = {
                "temperature": self.temperature,
                "top_p": self.top_p,
                "top_k": self.top_k,
                "max_output_tokens": self.max_output_tokens,
            }

            self.model = genai.GenerativeModel(model_name="gemini-pro-vision",generation_config=generation_config)
        except Exception as e:
            self.logger.error(f"Error setting up model: {e}")
            raise

    def generate_content(self, contents):
        self.logger.info(f"Generating contents")
        
        # Check model and contents for errors.
        if self.model is None:
            self.logger.error("Model is not initialized")
            raise ValueError("Model is not initialized")

        if contents is None:
            self.logger.error("Contents is not initialized")
            raise ValueError("Contents is not initialized")
        
        # Print out the contents list for debugging
        self.logger.info(f"Contents: {contents}")
        
        return self.model.generate_content(contents=contents)
