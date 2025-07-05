# app.py
import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
from av import VideoFrame

# Initialize MediaPipe Hands with lighter settings
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, model_complexity=0)
mp_draw = mp.solutions.drawing_utils

# Load TFLite model once
interpreter = tf.lite.Interpreter(model_path='data/sign_language_model.tflite')
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load labels
labels = np.load('data/label_encoder.npy', allow_pickle=True)

class SignLanguageProcessor(VideoProcessorBase):
    def __init__(self):
        self.result = "Waiting for detection..."
        self.no_hand_counter = 0
        self.frame_count = 0

    def preprocess_landmarks(self, landmarks):
        if not landmarks:
            return None
        data = []
        for lm in landmarks.landmark:
            data.extend([lm.x, lm.y, lm.z])
        return np.array(data).reshape(1, -1)

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.resize(img, (320, 240))  # Lower resolution
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            self.no_hand_counter = 0
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                data = self.preprocess_landmarks(hand_landmarks)
                if data is not None:
                    self.frame_count += 1
                    if self.frame_count % 5 == 0:  # Predict every 5th frame
                        input_data = data.astype(np.float32)
                        interpreter.set_tensor(input_details[0]['index'], input_data)
                        interpreter.invoke()
                        prediction = interpreter.get_tensor(output_details[0]['index'])
                        predicted_class = np.argmax(prediction)
                        confidence = prediction[0][predicted_class]
                        if confidence > 0.7:
                            gesture = labels[predicted_class]
                            self.result = f"Detected: {gesture} ({confidence:.2f})"
                            cv2.putText(img, f"{gesture} ({confidence:.2f})", (50, 50),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                            break
        else:
            self.no_hand_counter += 1
            if self.no_hand_counter > 30:
                self.result = "No hands detected"

        return VideoFrame.from_ndarray(img, format="bgr24")

def main():
    st.title("Real-Time Sign Language Interpreter")
    st.write("This app detects sign language gestures using your webcam.")

    RTC_CONFIGURATION = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )

    ctx = webrtc_streamer(
        key="sign-language",
        video_processor_factory=SignLanguageProcessor,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False}
    )

    if ctx.video_processor:
        st.write(ctx.video_processor.result)

if __name__ == "__main__":
    main()