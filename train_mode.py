import os
import math
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

from data_preprocessing import load_data, fix_center_paths, balance_data
from utils import batch_generator


def build_model():
    model = Sequential([
        Input(shape=(66, 200, 3)),

        Conv2D(24, (5, 5), strides=(2, 2), activation='relu'),
        Conv2D(36, (5, 5), strides=(2, 2), activation='relu'),
        Conv2D(48, (5, 5), strides=(2, 2), activation='relu'),
        Conv2D(64, (3, 3), activation='relu'),
        Conv2D(64, (3, 3), activation='relu'),

        Flatten(),
        Dense(100, activation='relu'),
        Dropout(0.5),
        Dense(50, activation='relu'),
        Dense(10, activation='relu'),
        Dense(1)
    ])

    model.compile(optimizer=Adam(learning_rate=0.0001), loss='mse')
    return model


if __name__ == "__main__":
    csv_path = "collected_data/driving_log.csv"
    img_folder = "collected_data/IMG"

    if not os.path.exists(csv_path):
        print("driving_log.csv not found.")
        exit()

    data = load_data(csv_path)
    data = fix_center_paths(data, img_folder)
    data = balance_data(data)

    image_paths = data['center'].values
    steering_values = data['steering'].astype(float).values

    X_train, X_valid, y_train, y_valid = train_test_split(
        image_paths,
        steering_values,
        test_size=0.2,
        random_state=42
    )

    print("Training samples:", len(X_train))
    print("Validation samples:", len(X_valid))

    batch_size = 32
    epochs = 15

    train_steps = max(1, math.ceil(len(X_train) / batch_size))
    valid_steps = max(1, math.ceil(len(X_valid) / batch_size))

    model = build_model()

    history = model.fit(
        batch_generator(X_train, y_train, batch_size, is_training=True),
        steps_per_epoch=train_steps,
        epochs=epochs,
        validation_data=batch_generator(X_valid, y_valid, batch_size, is_training=False),
        validation_steps=valid_steps,
        verbose=1
    )

    model.save("model.h5")
    print("Model saved as model.h5")

    plt.figure(figsize=(8, 5))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title("Training History")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig("training_history.png")
    plt.show()