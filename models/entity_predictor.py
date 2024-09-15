import sys
import os

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import tensorflow as tf
from tensorflow.keras import layers, models
from data_preparation import prepare_data  # Adjusted import statement

def create_model():
    """
    Create a Convolutional Neural Network (CNN) model.
    
    Returns:
    - model: Compiled Keras model.
    """
    model = models.Sequential()
    model.add(layers.Input(shape=(128, 128, 3)))  # Use Input layer
    model.add(layers.Conv2D(32, (3, 3), activation='relu'))
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

def train_model(batch_size=32, epochs=10):
    """
    Train the CNN model using the prepared data in batches.
    
    Parameters:
    - batch_size: Number of samples per gradient update.
    - epochs: Number of epochs to train the model.
    """
    # Prepare data
    X_train, X_val, y_train, y_val = prepare_data(batch_size)
    
    model = create_model()
    
    # Fit the model using the data
    model.fit(X_train, y_train, 
              validation_data=(X_val, y_val), 
              epochs=epochs, 
              batch_size=batch_size)
    
    model.save('models/entity_model.h5')  # Save the model for later use

if __name__ == "__main__":
    train_model(batch_size=32, epochs=10)