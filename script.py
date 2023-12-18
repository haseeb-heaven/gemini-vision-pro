"""
Description: This is the amazing Google Gemini Vision Pro.
This scans the image and using Gemini AI pro vision API it generates the descrption of the image.
It also uses the speech to text and text to speech to speak the prompt and display the description of the image.

Author: HeavenHM
Date: 17-12-2023
Version: 1.0
"""

import os
import sys
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import cv2
import io
from PIL import Image
from io import BytesIO
from pathlib import Path
import traceback
from libs.logger import Logger
from libs.gemini_vision import GeminiVision
from libs.speech import SpeechToText
from libs.voice import TextToSpeech
from libs.image_cv2 import ImageCV2, ImageTransformer
import subprocess

# Initialize session state
def init_session_state():
    if 'api_key' not in st.session_state:
        st.session_state['api_key'] = ''
    if 'temperature' not in st.session_state:
        st.session_state['temperature'] = 0.1
    if 'top_k' not in st.session_state:
        st.session_state['top_k'] = 32
    if 'top_p' not in st.session_state:
        st.session_state['top_p'] = 1.0
    if 'captured_image' not in st.session_state:
        st.session_state['captured_image'] = None
    if 'prompt' not in st.session_state:
        st.session_state['prompt'] = ''
    if 'api_key' not in st.session_state:
        st.session_state['api_key'] = ''
    if 'captured_image' not in st.session_state:
        st.session_state['captured_image'] = None
    if 'prompt' not in st.session_state:
        st.session_state['prompt'] = ''
    if "logger" not in st.session_state:
        st.session_state["logger"] = None
    if "tts" not in st.session_state:
        st.session_state["tts"] = None
    if "stt" not in st.session_state:
        st.session_state["stt"] = None
    if "gemini_vision" not in st.session_state:
        st.session_state["gemini_vision"] = None
    if "webrtc_ctx" not in st.session_state:
        st.session_state["webrtc_ctx"] = None
    if "response" not in st.session_state:
        st.session_state["response"] = None
    if "imageCv2" not in st.session_state:
        st.session_state["imageCv2"] = None

def display_support():
    st.markdown("<div style='text-align: center;'>Share and Support</div>", unsafe_allow_html=True)
    
    st.write("""
        <div style="display: flex; flex-direction: column; align-items: center; justify-content: center;">
            <ul style="list-style-type: none; margin: 0; padding: 0; display: flex;">
                <li style="margin-right: 10px;"><a href="https://twitter.com/haseeb_heaven" target="_blank"><img src="https://img.icons8.com/color/32/000000/twitter--v1.png"/></a></li>
                <li style="margin-right: 10px;"><a href="https://www.buymeacoffee.com/haseebheaven" target="_blank"><img src="https://img.icons8.com/color/32/000000/coffee-to-go--v1.png"/></a></li>
                <li style="margin-right: 10px;"><a href="https://www.youtube.com/@HaseebHeaven/videos" target="_blank"><img src="https://img.icons8.com/color/32/000000/youtube-play.png"/></a></li>
                <li><a href="https://github.com/haseeb-heaven/LangChain-Coder" target="_blank"><img src="https://img.icons8.com/color/32/000000/github--v1.png"/></a></li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

# Exception handling decorator
def exception_handler(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as exception:
            st.session_state.logger.error(f"An error occurred in {func.__name__}: {exception}")
            st.error(f"An error occurred: {exception}")
            st.session_state.logger.error(traceback.format_exc())
            st.stop()
    return wrapper

@exception_handler
def validate_image(image_path):
    if not image_path.exists():
        st.session_state.logger.error(f"Could not find image: {image_path}")
        raise FileNotFoundError(f"Could not find image: {image_path}")

@exception_handler
def process_image():    
    image_contents = [st.session_state['prompt'], st.session_state['captured_image']]
    st.session_state.logger.info(f"Image data is: {st.session_state['captured_image']}")
    
    response = st.session_state.gemini_vision.generate_content(image_contents)
    
    if 'error' in response:
        raise ValueError(f"An error occurred: {response}")
    else:
        if response.text:
            st.session_state.tts.speak(response.text)
            st.session_state.logger.info(f"Response: {response.text}")
            st.session_state.response = response.text
            
@exception_handler
def get_prompt_from_mic():
    prompt = st.session_state.stt.listen_and_convert()
    return prompt

@exception_handler
def capture_image(ctx):
    webcam_file_path = "webcam_snapshot.png"
        
    st.session_state.logger.info("Attempting to capture image from webcam with ImageCV2...")
    
    if st.session_state.imageCv2 is None:
        st.session_state.imageCv2 = ImageCV2()
    
    if ctx.video_transformer is None or ctx.video_transformer.in_image is None:
        st.error("Cannot capture image. Make sure your webcam is connected and not being used by another application.")
        return None
    
    image = st.session_state.imageCv2 .capture_image_from_webcam(ctx, webcam_file_path)
    
    return Image.open(image)

# Streamlit App
def streamlit_app():    
    
    # Google Logo and Title
    st.write('<div style="display: flex; flex-direction: row; align-items: center; justify-content: center;"><a style="margin-right: 10px;" href="https://www.google.com" target="_blank"><img src="https://img.icons8.com/color/32/000000/google-logo.png"/></a><h1 style="margin-left: 10px;">Google - Gemini Vision</h1></div>', unsafe_allow_html=True)
    
    # Display support
    display_support()
    
    # Initialize logger
    if st.session_state.logger is None:
        st.session_state.logger = Logger.get_logger('gemini_vision_pro.log')
        
    # Display the Gemini Sidebar settings
    with st.sidebar.title("Gemini Settings"):
        st.session_state.api_key = st.sidebar.text_input("API Key", type="password")
        st.session_state.temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.3)
        st.session_state.top_k = st.sidebar.number_input("Top K", value=32)
        st.session_state.top_p = st.sidebar.slider("Top P", 0.0, 1.0, 1.0)
        st.session_state.gemini_vision = GeminiVision(st.session_state.api_key, st.session_state.temperature, st.session_state.top_p, st.session_state.top_k)

        if (st.session_state.api_key is not None and st.session_state.api_key != '') \
            and (st.session_state.temperature is not None and st.session_state.temperature != '') \
            and (st.session_state.top_k is not None and st.session_state.top_k != '') \
            and (st.session_state.top_p is not None and st.session_state.top_p != ''):
            st.toast("Settings updated successfully!", icon="👍")
        else:
            st.toast("Please enter all the settings.\nAPI Key is required", icon="❌")
            raise ValueError("Please enter all values the settings.\nAPI Key is required")
        
    # Initialize services once
    if st.session_state.tts is None:
        st.session_state.tts = TextToSpeech()
    if st.session_state.stt is None:
        st.session_state.stt = SpeechToText()
    if st.session_state.gemini_vision is None:
        st.session_state.gemini_vision = GeminiVision(api_key=st.session_state['api_key'],
                                                      temperature=st.session_state['temperature'],
                                                      top_p=st.session_state['top_p'],
                                                      top_k=st.session_state['top_k'])
    
    # Webcam for capturing images
    st.title("Webcam")
    capture_ctx = webrtc_streamer(key="snapshot", video_transformer_factory=ImageTransformer)

    col1, col2,_,col3, = st.columns(4)

    with col1:
        if st.button("Capture Image"):
            st.session_state['captured_image'] = capture_image(capture_ctx)

    # Main Page
    with col2:
        if st.button("Speak Prompt"):
            
            # Validate API Key
            if st.session_state.api_key is None or st.session_state.api_key == '':
                st.toast("Please enter API Key in the sidebar.", icon="❌")
            else:
                st.session_state['prompt'] = get_prompt_from_mic()
            
    with col3:
        if st.button("Ask Gemini") and st.session_state['captured_image'] is not None:
            
            # Validate API Key
            if st.session_state.api_key is None or st.session_state.api_key == '':
                st.toast("Please enter API Key in the sidebar.", icon="❌")
            elif st.session_state.prompt is None or st.session_state.prompt == '':
                st.toast("Please enter prompt first.", icon="❌")
            else:
                # Check if image is captured
                if st.session_state['captured_image'] is None:
                    st.toast("Please capture image first.", icon="❌")
                else:
                    process_image()
    
    prompt = st.text_area(placeholder="Prompt:",label="Prompt",label_visibility="hidden",height=10,value=st.session_state.get('prompt',st.session_state['prompt']))
    st.session_state['prompt'] = prompt  # Update session state
    
    # if image is captured then display it
    if st.session_state['captured_image'] is not None:
        st.image(st.session_state['captured_image'], caption="Captured Image", use_column_width=True)
    
    # if response is present then display it
    if 'response' in st.session_state:
        st.code(f"Gemini AI: {st.session_state['response']}", language="python")
        
# create method to update pip using subprocess
def update_pip():
    subprocess.check_call(["python", "-m", "pip", "install", "--upgrade", "pip"])

# Main method
if __name__ == "__main__":
    try:
        init_session_state()
        update_pip()
        streamlit_app()
        
    except Exception as exception:
        import traceback
        if st.session_state.logger is None:
            print(f"An error occurred: {exception}")
            print(traceback.format_exc())
            st.error(f"An error occurred: {exception}")
            st.error(traceback.format_exc())
        else:
            st.session_state.logger.error(f"An error occurred: {exception}")
            st.session_state.logger.error(traceback.format_exc())
            st.error(f"An error occurred: {exception}")
