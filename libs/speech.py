import speech_recognition as sr
import logging

class SpeechToText:
    """
    A class that represents a speech-to-text converter.
    """

    def __init__(self):
        """
        Initialize the recognizer and the microphone.
        """
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)

    def listen_and_convert(self):
        """
        Listen to the microphone and convert the speech to text.
        """
        try:
            self.logger.info("Listening to the microphone...")
            with self.microphone as source:
                audio = self.recognizer.listen(source)
            self.logger.info("Converting speech to text...")
            text = self.recognizer.recognize_google(audio)
            self.logger.info(f"Converted text: {text}")
            return text
        except sr.UnknownValueError:
            self.logger.error("Google Speech Recognition could not understand the audio")
        except sr.RequestError as e:
            self.logger.error(f"Could not request results from Google Speech Recognition service: {str(e)}")