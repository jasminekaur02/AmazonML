# src/models/entity_predictor.py
import os
import tensorflow as tf
from tensorflow.keras import layers, models
import sys
import os

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from data_preparation import prepare_data  # Now this should work


def create_model():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(1))  # Adjust output layer based on the number of entities
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    return model

def train_model():
    X_train, X_val, y_train, y_val = prepare_data()
    model = create_model()
    model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))
    model.save('models/entity_model.h5')  # Save the model for later use

if __name__ == "__main__":
    train_model()