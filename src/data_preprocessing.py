import cv2
import mediapipe as mp
import numpy as np
import os

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            data = []
            for lm in hand_landmarks.landmark:
                data.extend([lm.x, lm.y, lm.z])
            return np.array(data)
    return None

def process_dataset(data_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    X, y = [], []
    for label in os.listdir(data_dir):
        label_dir = os.path.join(data_dir, label)
        if os.path.isdir(label_dir):
            for img_file in os.listdir(label_dir):
                img_path = os.path.join(label_dir, img_file)
                data = preprocess_image(img_path)
                if data is not None:
                    X.append(data)
                    y.append(label)
                    np.save(f"{output_dir}/{label}_{len(X)}.npy", data)
    np.save(f"{output_dir}/labels.npy", np.array(y))

if __name__ == "__main__":
    process_dataset('data/asl_alphabet_train/', 'data/processed')