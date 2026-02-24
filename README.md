# 🤟 AI Sign Language Teacher
An interactive, real-time machine learning application designed to teach users hand signs and gestures using computer vision.

---

## 🌟 Project Overview
This project aims to bridge the communication gap by providing an automated way for anyone to learn sign language. By leveraging **MediaPipe** for hand tracking and **TensorFlow/Keras** for gesture classification, the application provides instant feedback on a user's signs compared to a reference library.

## 🚀 Key Features
*   **Real-Time Detection**: Low-latency hand landmark tracking using [MediaPipe Hands](https://google.github.io).
*   **Interactive Lessons**: Visual guides showing the target sign for users to mimic.
*   **Instant Feedback**: Visual cues (green/red overlays) to signal correct or incorrect gestures.
*   **Progress Tracking**: A simple gamified system to track signs mastered over time.

## 🛠️ Tech Stack
*   **Language**: [Python 3.x](https://www.python.org)
*   **Computer Vision**: [OpenCV](https://opencv.org)
*   **Hand Tracking**: [MediaPipe](https://ai.google.dev)
*   **Machine Learning**: [TensorFlow](https://www.tensorflow.org) / [Scikit-learn](https://scikit-learn.org)
*   **UI/Dashboard**: [Streamlit](https://streamlit.io) (Planned)

## 📂 Project Structure
```text
├── data/               # Raw images and processed landmark CSVs
├── models/             # Trained model weights (.h5, .tflite)
├── notebooks/          # Exploratory Data Analysis & Prototyping
├── src/                # Core source code
│   ├── collection.py   # Webcam data capture script
│   ├── processing.py   # Landmark extraction & normalization
│   ├── train.py        # Model training logic
│   └── app.py          # Main application interface
├── requirements.txt    # Project dependencies
└── README.md           # You are here!
