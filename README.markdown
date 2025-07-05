# Real-Time Sign Language Interpreter

A Python-based project for real-time sign language interpretation using MediaPipe for hand gesture detection and a neural network for gesture classification. Uses the ASL Alphabet dataset from Kaggle.

## Prerequisites
- Python 3.8+
- Webcam for real-time testing
- Kaggle account to download the dataset

## Setup Instructions
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/sign_language_interpreter.git
   cd sign_language_interpreter
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the ASL Alphabet Dataset**:
   - Go to [Kaggle ASL Alphabet Dataset](https://www.kaggle.com/datasets/grassknoted/asl-alphabet).
   - Download and extract `asl_alphabet_train` to the `data/` directory.
   - Alternatively, use the Kaggle API:
     ```bash
     kaggle datasets download -d grassknoted/asl-alphabet -p data/
     unzip data/asl-alphabet.zip -d data/asl_alphabet_train
     ```

4. **Preprocess the Dataset**:
   ```bash
   python src/data_preprocessing.py
   ```

5. **Train the Model**:
   ```bash
   python src/model_training.py
   ```

6. **Run the Interpreter**:
   ```bash
   python src/sign_language_interpreter.py
   ```

## Dataset
- **Source**: [ASL Alphabet Dataset](https://www.kaggle.com/datasets/grassknoted/asl-alphabet) (CC BY-SA 4.0)
- Contains images for 29 classes (A-Z, space, delete, nothing).

## Notes
- The project uses MediaPipe for hand landmark detection and a neural network for classification.
- Adjust the confidence threshold in `sign_language_interpreter.py` for better performance.
- The trained model (`sign_language_model.h5`) is not included due to size. Train it using the provided script.

## License
This project is licensed under the MIT License. The ASL Alphabet dataset is licensed under CC BY-SA 4.0.