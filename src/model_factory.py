import os

from tensorflow.keras import layers, models


class model_factory:
    def __init__(self):
        self.num_classes = len(
            [
                d.name
                for d in os.scandir(r"C:\Users\repla\signLearn\data")
                if d.is_dir() and not d.name.startswith(".")
            ]
        )

    def create_model(self):
        model = models.Sequential(
            [
                # Input: 21 landmarks * 2 coordinates (x, y) = 42 features
                layers.Input(shape=(42, 1)),
                layers.Conv1D(64, kernel_size=3, activation="relu"),
                layers.Conv1D(128, kernel_size=3, activation="relu"),
                layers.MaxPooling1D(pool_size=2),
                layers.Dropout(0.3),
                layers.Flatten(),
                layers.Dense(128, activation="relu"),
                layers.Dropout(0.5),
                layers.Dense(self.num_classes, activation="softmax"),
            ]
        )

        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        return model
