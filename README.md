# GestAI - Decoding Human Gestures in Real Time

This repository contains Python code that implements a real-time action recognition system using deep learning and OpenCV. The system captures video from a webcam, detects human body landmarks using MediaPipe's Holistic model, extracts sequences of keypoints from these landmarks, and uses a trained LSTM (Long Short-Term Memory) deep learning model to classify the actions depicted in the sequences.

## Table of Contents
- [Dependencies](#dependencies)
- [Installation](#installation)
- [Usage](#usage)
- [Model Training](#model-training)
- [Real-Time Gesture Recognition](#real-time-gesture-recognition)
- [License](#license)

## Dependencies
- OpenCV
- MediaPipe
- TensorFlow
- scikit-learn
- NumPy
- Matplotlib
- PIL
- SciPy
- Tkinter

## Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/gesture-recognition.git
    ```
2. Navigate to the project directory:
    ```bash
    cd gesture-recognition
    ```
3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage
1. Run the provided code in a Python environment. Ensure your webcam is accessible and not being used by other applications.
2. Perform gestures in front of the webcam as per the predefined set of gestures.
3. The program will identify and display the recognized gestures in real-time.

## Model Training
1. Data Collection:
   - Adjust the `DATA_PATH`, `actions`, `sequence_count`, and `sequence_length` variables to match your data collection setup.
   - Run the data collection section of the code to gather training data.
2. Data Preprocessing:
   - The data preprocessing section will organize the collected data for training.
3. Neural Network Training:
   - Run the model training section of the code to train the LSTM neural network.
   - The trained model will be saved as `action.h5` in the project directory.

## Real-Time Gesture Recognition
1. Load the trained model `action.h5`.
2. Run the real-time gesture recognition section of the code.
3. Perform gestures in front of the webcam, and the program will display the recognized gestures along with the prediction confidence.

## License
This project is licensed under the MIT License - see the [LICENSE.md](https://github.com/vbabua/GestAI-Decoding-Human-Gestures-in-Real-Time/blob/main/LICENSE) file for details.
