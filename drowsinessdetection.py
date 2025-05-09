import cv2
import os
import numpy as np
import av
import threading
import queue
from keras.models import load_model
from pygame import mixer
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration

# Initialize mixer for alarm
mixer.init()
alarm_sound = mixer.Sound('alarm.wav')

# Load Haar cascades and CNN model
face_cascade = cv2.CascadeClassifier('haar cascade files/haarcascade_frontalface_alt.xml')
leye_cascade = cv2.CascadeClassifier('haar cascade files/haarcascade_lefteye_2splits.xml')
reye_cascade = cv2.CascadeClassifier('haar cascade files/haarcascade_righteye_2splits.xml')
model = load_model('models/cnncat2.h5')

# Streamlit UI setup
st.title("🚗 SafeDrive AI: Drowsiness Detection")
threshold = st.sidebar.slider("Drowsiness Threshold", min_value=5, max_value=50, value=15)
show_score = st.sidebar.checkbox("Display Drowsiness Score", value=True)

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
}
)

class DrowsinessTransformer(VideoTransformerBase):
    def __init__(self):
        self.score = 0
        self.thicc = 2

    def transform(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        height, width = img.shape[:2]

        # Detection
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)
        left_eye = leye_cascade.detectMultiScale(gray)
        right_eye = reye_cascade.detectMultiScale(gray)

        # Process eyes
        rpred = lpred = [1]
        for (x, y, w, h) in right_eye:
            roi = gray[y:y+h, x:x+w]
            roi = cv2.resize(roi, (24, 24)) / 255.0
            roi = roi.reshape(1, 24, 24, 1)
            rpred = np.argmax(model.predict(roi), axis=-1)
            break
        for (x, y, w, h) in left_eye:
            roi = gray[y:y+h, x:x+w]
            roi = cv2.resize(roi, (24, 24)) / 255.0
            roi = roi.reshape(1, 24, 24, 1)
            lpred = np.argmax(model.predict(roi), axis=-1)
            break

        # Update score
        if rpred[0] == 0 and lpred[0] == 0:
            self.score += 1
        else:
            self.score = max(self.score - 1, 0)

        # Draw
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (100, 100, 100), 1)
        if self.score > threshold:
            try:
                alarm_sound.play()
            except:
                pass
            cv2.rectangle(img, (0, 0), (width, height), (0, 0, 255), self.thicc)
            self.thicc = 2 if self.thicc >= 16 else self.thicc + 2

        if show_score:
            cv2.putText(img, f"Score: {self.score}", (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

webrtc_streamer(
    key="drowsiness",
    video_transformer_factory=DrowsinessTransformer,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
    async_transform=True
)
