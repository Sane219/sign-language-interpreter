import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import LabelEncoder

def load_data(data_dir):
    X, y = [], []
    labels = np.load(f"{data_dir}/labels.npy", allow_pickle=True)
    for file in os.listdir(data_dir):
        if file.endswith('.npy') and file != 'labels.npy':
            data = np.load(os.path.join(data_dir, file))
            X.append(data)
    return np.array(X), labels

def train_model(data_dir, save_path):
    X, y = load_data(data_dir)
    le = LabelEncoder()
    y = le.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = Sequential([
        Dense(128, activation='relu', input_shape=(X.shape[1],)),
        Dense(64, activation='relu'),
        Dense(len(le.classes_), activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))
    model.save(save_path)
    np.save('data/label_encoder.npy', le.classes_)

if __name__ == "__main__":
    train_model('data/processed', 'data/sign_language_model.h5')