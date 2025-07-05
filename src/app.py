import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils
model = load_model('data/sign_language_model.h5')
labels = np.load('data/label_encoder.npy', allow_pickle=True)

class SignLanguageProcessor(VideoProcessorBase):
    def __init__(self):
        self.result = None

    def preprocess_landmarks(self, landmarks):
        if not landmarks:
            return None
        data = []
        for lm in landmarks.landmark:
            data.extend([lm.x, lm.y, lm.z])
        return np.array(data).reshape(1, -1)

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                data = self.preprocess_landmarks(hand_landmarks)
                if data is not None:
                    prediction = model.predict(data, verbose=0)
                    predicted_class = np.argmax(prediction)
                    confidence = prediction[0][predicted_class]
                    if confidence > 0.7:
                        gesture = labels[predicted_class]
                        cv2.putText(img, f"{gesture} ({confidence:.2f})", (50, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        self.result = f"Detected: {gesture} ({confidence:.2f})"
                    else:
                        self.result = "No confident detection"
                else:
                    self.result = "No landmarks detected"
        else:
            self.result = "No hands detected"

        return img

def main():
    st.title("Real-Time Sign Language Interpreter")
    st.write("This app detects sign language gestures using your webcam.")
    
    # WebRTC configuration for Streamlit
    RTC_CONFIGURATION = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )

    # Start webcam stream
    ctx = webrtc_streamer(
        key="sign-language",
        video_processor_factory=SignLanguageProcessor,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False}
    )

    # Display detection result
    if ctx.video_processor:
        st.write(ctx.video_processor.result or "Waiting for detection...")

if __name__ == "__main__":
    main()