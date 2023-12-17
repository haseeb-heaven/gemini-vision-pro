from gtts import gTTS
import os
import logging

class TextToSpeech:
    """
    A class that represents a text-to-speech converter.
    """

    def __init__(self):
        """
        Initialize the logger.
        """
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)

    def speak(self, text):
        """
        Convert the given text to speech.
        """
        try:
            self.logger.info(f"Speaking the text: {text}")
            tts = gTTS(text=text, lang='en')
            tts.save("speech.mp3")
            os.system("mpg321 speech.mp3")
            os.remove("speech.mp3")
        except Exception as exception:
            self.logger.error(f"An error occurred while trying to speak the text: {str(exception)}")
            raise