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
from libs.image_cv2 import ImageCV2

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
    #st.session_state['prompt'] = f"{st.session_state['prompt']}, In just 20 words be very concise and clear"
    image_prompt = st.session_state['prompt']
    web_image = st.session_state['captured_image']
    st.info(f"Type of web_image: {type(web_image)}")
    st.info(f"Data of web_image: {web_image}")
    
    image_contents = [st.session_state['prompt'], st.session_state['captured_image']]
    st.session_state.logger.info(f"Image data is: {st.session_state['captured_image']}")
    
    response = st.session_state.gemini_vision.generate_content(image_contents)
    
    if 'error' in response:
        raise ValueError(f"An error occurred: {response}")
    else:
        if response.text:
            st.session_state.tts.speak(response.text)
            st.markdown(response.text)
            
@exception_handler
def get_prompt_from_mic():
    prompt = st.session_state.stt.listen_and_convert()
    return prompt

@exception_handler
def log_webrtc_context_states(webrtc_ctx):
    if webrtc_ctx is not None:
        # Log the state of the WebRTC context
        st.info(f"WebRTC context: {webrtc_ctx}")
        st.info(f"Is WebRTC playing: {webrtc_ctx.state.playing}")
        st.info(f"Is audio receiver ready: {webrtc_ctx.audio_receiver}")
        st.info(f"Is video receiver ready: {webrtc_ctx.video_receiver}")
    else:
        st.error("WebRTC context is None.")


@exception_handler
def capture_image():
    st.info("Attempting to capture image from webcam with ImageCV2...")
    
    # Capture the image from the webcam
    web_image = None
    web_cam = ImageCV2()
 
    web_image_file = "web_image.png"
    web_image = web_cam.capture_image_from_webcam(web_image_file)
    if web_image is None:
        raise ValueError("Could not capture image from webcam")
    
    # convert web_image from RGB to RGBA
    web_image = web_image.convert("RGBA")
    
    # Validate that an image is present
    image_path = Path(web_image_file)
    validate_image(image_path)
    
    # Open the image
    st.session_state.logger.info(f"Trying to open image: {web_image_file}")
    web_image = Image.open(web_image_file)
    return web_image

# Streamlit App
def streamlit_app():
    st.title("Gemini Vision Streamlit App")
    
    # Initialize logger and services once
    if st.session_state.logger is None:
        st.session_state.logger = Logger.get_logger('gemini_vision_pro.log')
    if st.session_state.tts is None:
        st.session_state.tts = TextToSpeech()
    if st.session_state.stt is None:
        st.session_state.stt = SpeechToText()
    if st.session_state.gemini_vision is None:
        st.session_state.gemini_vision = GeminiVision(api_key=st.session_state['api_key'],
                                                      temperature=st.session_state['temperature'],
                                                      top_p=st.session_state['top_p'],
                                                      top_k=st.session_state['top_k'])
    

    # Main Page
    enable_mic = st.checkbox("Enable Microphone")
    if enable_mic and st.button("Get Prompt from Mic"):
        st.session_state['prompt'] = get_prompt_from_mic()

    prompt = st.text_input("Enter prompt", value=st.session_state.get('prompt', ''))
    st.session_state['prompt'] = prompt  # Update session state

    # WebRTC streamer
    webrtc_ctx = webrtc_streamer(
                key="webcam", 
                mode=WebRtcMode.SENDRECV, 
                rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
                video_frame_callback=lambda frame: None
            )
   
    capture_button = st.button("Capture Image")
    send_button = st.button("Gemini Vision")

    if capture_button:
        st.session_state['captured_image'] = capture_image()
        if st.session_state['captured_image'] is not None:
            st.image(st.session_state['captured_image'], width=300, caption="Captured Image")
            st.toast("Image captured successfully!")
        else:
            st.warning("Failed to capture image. Please try again.")
        
    if send_button and st.session_state['captured_image'] is not None:
        process_image()
        
    elif send_button:
        st.warning("Please capture an image first.")
        
    # # Configure Gemini Vision settings from the sidebar
    with st.sidebar.title("Settings"):
        st.session_state.api_key = st.sidebar.text_input("API Key", type="password")
        st.session_state.temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.9)
        st.session_state.top_k = st.sidebar.number_input("Top K", value=1)
        st.session_state.top_p = st.sidebar.slider("Top P", 0.0, 1.0, 1.0)
        st.session_state.gemini_vision = GeminiVision(st.session_state.api_key, st.session_state.temperature, st.session_state.top_k, st.session_state.top_p)

        st.toast("Settings updated successfully!")
        
        
if __name__ == "__main__":
    try:
        init_session_state()
        streamlit_app()
    except Exception as exception:
        import traceback
        st.session_state.logger.error(f"An error occurred: {exception}")
        st.session_state.logger.error(traceback.format_exc())
        st.error(f"An error occurred: {exception}")
