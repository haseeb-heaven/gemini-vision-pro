import time
import cv2
from libs.logger import Logger
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer

class ImageTransformer(VideoTransformerBase):
    def __init__(self):
        self.in_image = None

    def transform(self, frame):
        self.in_image = frame.to_ndarray(format="bgr24")
        return self.in_image

class ImageCV2:
    # Assuming other parts of the class are defined above

    def __init__(self):
        self.logger = Logger.get_logger("gemini_vision_pro.txt")
        self.logger.info("Initializing ImageCV2")
    
    def capture_image_from_webcam(self, ctx, image_name):
        self.logger.info("Capturing image from webcam")

        # Check if the webcam context is initialized
        if ctx is None:
            self.logger.error("Webcam context is None")
            raise ValueError("Webcam context is not initialized")

        # Check if the video transformer and the image in the transformer are available
        if ctx.video_transformer is None or ctx.video_transformer.in_image is None:
            self.logger.error("Cannot capture image")
            raise ValueError("Cannot capture image from webcam")

        # Check if the webcam is streaming
        if not ctx.state.playing:
            self.logger.warning("Webcam is not streaming")
            raise RuntimeError("Webcam is not streaming")

        # Capture and save the image
        image = ctx.video_transformer.in_image
        cv2.imwrite(image_name, image)
        self.logger.info(f"Saved image to: {image_name}")
        return image_name
