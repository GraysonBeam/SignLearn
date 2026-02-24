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

    # 5. Step C: Flattening
    # Convert [[x1,y1], [x2,y2]...] into [x1, y1, x2, y2...]
    return normalized_coords.flatten()
