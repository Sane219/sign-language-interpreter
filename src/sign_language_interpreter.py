import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils
model = load_model('data/sign_language_model.h5')
labels = np.load('data/label_encoder.npy', allow_pickle=True)

def preprocess_landmarks(landmarks):
    if not landmarks:
        return None
    data = []
    for lm in landmarks.landmark:
        data.extend([lm.x, lm.y, lm.z])
    return np.array(data).reshape(1, -1)

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                data = preprocess_landmarks(hand_landmarks)
                if data is not None:
                    prediction = model.predict(data)
                    predicted_class = np.argmax(prediction)
                    confidence = prediction[0][predicted_class]
                    if confidence > 0.7:
                        gesture = labels[predicted_class]
                        cv2.putText(frame, f"{gesture} ({confidence:.2f})", (50, 50),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Sign Language Interpreter', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    hands.close()

if __name__ == "__main__":
    main()