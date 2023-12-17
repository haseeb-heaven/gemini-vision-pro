import time
import cv2
from libs.logger import Logger
from PIL import Image
import numpy as np

class ImageCV2:
    
    def __init__(self) -> None:
        # Set up logging
        self.logger = Logger.get_logger('gemini_vision.log')
        
    def open_webcam(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            self.logger.error("Cannot open webcam")
            return None
        return cap

    def capture_image(self, cap):
        ret, frame = cap.read()
        self.logger.info(f"Capturing image from webcam")
        
        if not ret:
            self.logger.error("Cannot capture image")
            return None

        self.logger.info(f"Converting image PIL.Image")
        # Convert the numpy.ndarray to a PIL.Image.Image
        image = Image.fromarray(frame)
        
        self.logger.info(f"Converting image success")
        return image
    
    def save_image(self, image, filename):
        self.logger.info(f"Saving image to: {filename}")
        
        # Convert the PIL.Image.Image back to a numpy.ndarray
        frame = np.array(image)
        
        # Save the image
        cv2.imwrite(filename, frame)
        
    def capture_image_from_webcam(self,image_name):
        self.logger.info(f"Capturing image from webcam")
        #time.sleep(5)
                
        cap = self.open_webcam()
        time.sleep(1)
        
        if cap is None:
            self.logger.error("Cannot open webcam")
            return None

        image = self.capture_image(cap)
        
        # Check if frame is None
        if image is None:
            self.logger.error("Cannot capture image")
            return None
        
        time.sleep(1)
        
        # Save the image
        self.save_image(image, image_name)
        self.logger.info(f"Saved image to: {image_name}")

        return image
    
    def show_webcam_feed(self):
        # Open the webcam (0 is the default webcam)
        cap = cv2.VideoCapture(0)

        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()

            # Display the resulting frame
            cv2.imshow('Webcam Feed', frame)

            # Break the loop on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # When everything is done, release the capture and destroy the window
        cap.release()
        cv2.destroyAllWindows()
        
    def stop_webcam_feed(self,interval):
        time.sleep(interval)

