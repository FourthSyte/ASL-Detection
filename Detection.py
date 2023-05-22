import streamlit as st
from PIL import Image
from streamlit_webrtc import webrtc_streamer
import av
import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math

st.set_page_config(
    page_title="Detection",
    layout='centered',
    page_icon='./images/sign-language.png'
)

st.title("Sign Language Detection")
st.caption("This web demonstrates Sign Language Detection")

detector = HandDetector(maxHands=2)
classifier = Classifier('Model/keras_model.h5', 'Model/labels.txt')

offset = 20
imgsize = 300
labels = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P",
    "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "Hello", "Yes", "No", "ILoveYou", "ThankYou",
    "Eat", "Drink", "Like", "Wrong"
]


def video_frame_callback(frame):
    try:
        img = frame.to_ndarray(format="bgr24")
        imgoutput = img.copy()
        hands, img = detector.findHands(img)

        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']
            imgwhite = np.ones((imgsize, imgsize, 3), np.uint8) * 255
            imgcrop = img[y - offset:y + h + offset, x - offset:x + w + offset]
            aspectratio = h / w

            if aspectratio > 1:
                k = imgsize / h
                wcal = math.ceil(k * w)
                imgresize = cv2.resize(imgcrop, (wcal, imgsize))
                wgap = math.ceil((imgsize - wcal) / 2)
                imgwhite[:, wgap:wcal + wgap] = imgresize
                prediction, index = classifier.getPrediction(imgwhite, draw=False)
            else:
                k = imgsize / w
                hcal = math.ceil(k * h)
                imgresize = cv2.resize(imgcrop, (imgsize, hcal))
                hgap = math.ceil((imgsize - hcal) / 2)
                imgwhite[hgap:hcal + hgap, :] = imgresize
                prediction, index = classifier.getPrediction(imgwhite, draw=False)

                cv2.putText(
                    imgoutput, labels[index], (x, y - 20),
                    cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 255), 2
                )
                cv2.rectangle(
                    imgoutput, (x - offset, y - offset),
                    (x + w + offset, y + h + offset), (255, 0, 255), 2
                )

        return av.VideoFrame.from_ndarray(imgoutput, format="bgr24")
    except Exception as e:
        print("An error occurred in the video_frame_callback function:", e)
        return None


webrtc_streamer(
    key="example",
    video_frame_callback=video_frame_callback,
    media_stream_constraints={"video": True, "audio": False}
)

with st.container():
    st.write("---")

st.markdown("""
### Reminder:

For accurate sign language detection, please ensure the following:

1. Position the gestures correctly within the camera frame.
2. Use a high-definition (HD/2k) camera for capturing clear and detailed visuals.
3. Ensure good lighting conditions in your environment to enhance visibility.
4. Maintain a clutter-free and distraction-free environment.

These steps will help optimize the accuracy and performance of the sign language detection feature. Thank you for your cooperation!
""")

with st.container():
    st.write("---")

image_paths = ['images/1.png', 'images/2.png', 'images/3.png', 'images/4.png', 'images/5.png', 'images/6.png',
               'images/7.png', 'images/8.png', 'images/9.png', 'images/10.png',
               'images/11.png', 'images/12.png', 'images/13.png', 'images/14.png', 'images/15.png', 'images/16.png',
               'images/17.png', 'images/18.png', 'images/19.png', 'images/20.png',
               'images/21.png', 'images/22.png', 'images/23.png', 'images/24.png', 'images/25.png', 'images/26.png',
               'images/27.png', 'images/28.png', 'images/29.png', 'images/30.png',
               'images/31.png', 'images/32.png', 'images/33.png', 'images/34.png', 'images/35.png', 'images/36.png']

st.title("Detectable gestures by the detection system")
st.caption("Presented below are a series of example gestures that can serve as a guide for effectively "
           "utilizing the system. "
           "This detection system is capable of recognizing alphabets and a selection of words.")

# Initialize slideshow index and total number of images
slideshow_index = st.session_state.get('slideshow_index', 0)
num_images = len(image_paths)


# Display the current image
def display_image(image_index):
    image_path = image_paths[image_index]
    image = Image.open(image_path)
    st.image(image, use_column_width=True)


# Create a layout for slideshow with navigation buttons
col1, col2, col3 = st.columns([1, 10, 1])

# Add navigation buttons
if col1.button('⬅️') and slideshow_index > 0:
    slideshow_index -= 1

with col3:
    if col3.button('➡️') and slideshow_index < num_images - 1:
        slideshow_index += 1

# Save the current index to session state
st.session_state['slideshow_index'] = slideshow_index

# Display the current image
display_image(slideshow_index)
