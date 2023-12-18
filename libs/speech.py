import os
import sounddevice as sd
import numpy as np
import whisper
import logging
import wavio
from libs.logger import Logger

class SpeechToText:
    """
    A class that represents a speech-to-text converter using OpenAI's Whisper.
    """

    def __init__(self, duration=5, fs=44100):
        self.model = whisper.load_model("base")
        self.duration = duration
        self.fs = fs
        self.logger = Logger.get_logger("gemini_vision_pro.log")

    def record_audio(self):
        """
        Record audio from the microphone.
        """
        self.logger.info("Recording audio...")
        recording = sd.rec(int(self.duration * self.fs), samplerate=self.fs, channels=1)
        sd.wait()
        return recording

    def listen_and_convert(self):
        """
        Convert the recorded audio to text using Whisper.
        """
        try:
            recording = self.record_audio()
            # Save the recording temporarily
            wavio.write("temp.wav", recording, self.fs, sampwidth=2)
            # Transcribe the audio file
            result = self.model.transcribe("temp.wav")
            text = result["text"]
            self.logger.info(f"Converted text: {text}")
            os.remove("temp.wav")
            return text
        except Exception as exception:
            self.logger.error(f"Error in Whisper speech recognition: {str(exception)}")
