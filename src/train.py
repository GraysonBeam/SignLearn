from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split


def start_training(model):
    BASE_PATH = Path(__file__).resolve().parent
    DATA_PATH = BASE_PATH.parent / "data"

    # 1. Load the master files
    X = np.load(DATA_PATH / "x_train.npy")
    y = np.load(DATA_PATH / "y_train.npy")

    # Reshape (Samples, 42) -> (Samples, 42, 1) for Conv1D
    X = X.reshape(X.shape[0], X.shape[1], 1)

    # 2. Split data: 80% to learn, 20% to test accuracy
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 2. Train
    print(f"Training on {len(X_train)} samples...")
    model.fit(
        X_train, y_train, epochs=150, batch_size=32, validation_data=(X_test, y_test)
    )

    # 7. Save the model for use in your app
    model.save("models/sign_language_alphabet_model.h5")
    print("Training complete. Model saved as models/sign_language_alphabet_model.h5")
