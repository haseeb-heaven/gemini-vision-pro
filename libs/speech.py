import deepspeech
import numpy as np
from libs.logger import Logger

class DeepSpeechToText:
    def __init__(self, model_path):
        self.model = deepspeech.Model(model_path)
        self.logger = Logger.get_logger("gemini_vision_pro.log")

    def listen_and_convert(self, audio):
        try:
            self.logger.info("Converting speech to text using DeepSpeech...")
            audio_np = np.frombuffer(audio, dtype=np.int16)
            text = self.model.stt(audio_np)
            self.logger.info(f"Converted text: {text}")
            return text
        except Exception as exception:
            self.logger.error(f"Error in DeepSpeech speech recognition: {str(exception)}")
