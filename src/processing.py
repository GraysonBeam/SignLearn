import json as json_module
import os
from pathlib import Path

import cv2
import mediapipe.python.solutions.hands as mp_hands
import numpy as np

# Initialize the engine
hands = mp_hands.Hands(
    static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5
)


def extract_normalized_landmarks(image_rgb):
    """
    Processes an image and returns a flat, normalized list of 21 (x, y) coordinates.
    """
    # 1. Detection Phase
    results = hands.process(image_rgb)

    if not results.multi_hand_landmarks:
        return None  # No hand found in the image

    # We only take the first hand detected
    hand_landmarks = results.multi_hand_landmarks[0]

    # 2. Coordinate Extraction
    # We create a list of [x, y] for all 21 points
    raw_coords = []
    for lm in hand_landmarks.landmark:
        raw_coords.append([lm.x, lm.y])

    # Convert to NumPy array for easy math
    coords = np.array(raw_coords)

    # 3. Step A: Translation (Zero-Centering at the Wrist)
    # The wrist is Landmark 0. We subtract it from everything.
    base_x, base_y = coords[0]
    normalized_coords = coords - [base_x, base_y]

    # 4. Step B: Scaling (Size Invariance)
    # We find the largest distance from the wrist to keep coordinates between -1 and 1
    max_value = np.max(np.abs(normalized_coords))
    if max_value != 0:
        normalized_coords = normalized_coords / max_value

    # 5. flatten the list to a single dimension (21 points * 2 coordinates = 42 features)
    normalized_coords = normalized_coords.flatten().tolist()

    return normalized_coords


def start_processing():
    BASE_DIR = Path(__file__).resolve().parent

    os.system("cls" if os.name == "nt" else "clear")
    print("Processing data...")
    print("[] 0%\r", end="")

    RAW_DATA_PATH = BASE_DIR.parent / "data" / "raw_data"
    DATA_PATH = BASE_DIR.parent / "data"

    # Use os.scandir to iterate through the main directory
    letter = 0
    total_features = []
    y_train = []

    for symbol_entry in os.scandir(RAW_DATA_PATH):
        # 1. Skip non-directory files like __init__.py or hidden files
        if not symbol_entry.is_dir() or symbol_entry.name.startswith("."):
            continue

        # 2. Iterate through images in the symbol's directory
        count = 0
        percent = 0
        for file_entry in os.scandir(f"{symbol_entry.path}"):
            # Process only files (images), skipping the 'processed_data' folder itself
            if file_entry.is_file() and not file_entry.name.startswith("."):
                # Load image with OpenCV to pass RGB data to MediaPipe
                image = cv2.imread(file_entry.path)
                if image is None:
                    continue
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                # Extract landmarks
                if extract_normalized_landmarks(image_rgb) is None:
                    continue
                coords = extract_normalized_landmarks(image_rgb)
                total_features.append(coords)
                y_train.append(letter)
                count += 1

                if count == int(len(os.listdir(f"{symbol_entry.path}")) / 100):
                    count = 0
                    percent += 1
                    os.system("cls" if os.name == "nt" else "clear")
                    print("Processing data for symbol: " + symbol_entry.name)
                    print(f"[{'#' * percent}] {percent}%\r", end="")

        letter += 1

    x_train_filename = os.path.join(DATA_PATH, "x_train.npy")
    y_train_filename = os.path.join(DATA_PATH, "y_train.npy")

    if total_features:
        np.save(x_train_filename, np.array(total_features))
        np.save(y_train_filename, np.array(y_train))

    # Create the translation map: {0: "A", 1: "B", ...}
    symbols = [
        d.name
        for d in os.scandir(DATA_PATH)
        if d.is_dir() and not d.name.startswith(".")
    ]
    label_map = {i: name for i, name in enumerate(symbols)}

    # Save it so Streamlit app can load it later
    with open(os.path.join(DATA_PATH, "label_map.json"), "w") as f:
        json_module.dump(label_map, f)
