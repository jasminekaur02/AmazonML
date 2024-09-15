import pandas as pd
import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split

def convert_label_to_float(label):
    """
    Convert various measurement formats to a float.
    
    Parameters:
    - label: The label string to convert.

    Returns:
    - float value of the label or None if conversion fails.
    """
    try:
        # Remove any non-numeric characters except for decimal points
        numeric_value = ''.join(filter(lambda x: x.isdigit() or x in ['.', ' '], label)).strip()
        if ' ' in numeric_value:
            # If there are multiple numbers, take the first one
            numeric_value = numeric_value.split()[0]
        return float(numeric_value)
    except ValueError:
        return None

def load_data_in_batches(image_dir, csv_file, batch_size):
    """
    Load images and labels from the specified CSV file in batches.

    Parameters:
    - image_dir: Directory where images are stored.
    - csv_file: Path to the CSV file containing image links and labels.
    - batch_size: Number of samples to load in each batch.

    Yields:
    - A tuple of (images, labels) for each batch.
    """
    df = pd.read_csv(csv_file)
    total_samples = len(df)
    
    for start in range(0, total_samples, batch_size):
        end = min(start + batch_size, total_samples)
        images = []
        labels = []
        
        for index in range(start, end):
            img_path = os.path.join(image_dir, df['image_link'][index].split('/')[-1])
            try:
                image = cv2.imread(img_path)
                if image is None:
                    raise ValueError(f"Image not found or cannot be loaded: {img_path}")
                
                image = cv2.resize(image, (128, 128))  # Resize to fit model input
                images.append(image)
                
                # Convert the label to a numeric type
                label = df['entity_value'][index]
                numeric_label = convert_label_to_float(label)
                if numeric_label is not None:
                    labels.append(numeric_label)
                else:
                    print(f"Warning: Unable to convert label '{label}' to float. Skipping this entry.")
            except Exception as e:
                print(f"Error processing {img_path}: {e}. Skipping this entry.")
        
        yield np.array(images), np.array(labels)

def prepare_data(batch_size=32):
    """
    Prepare training and validation data in batches.

    Returns:
    - X_train: Training images.
    - X_val: Validation images.
    - y_train: Training labels.
    - y_val: Validation labels.
    """
    images, labels = [], []
    
    for batch_images, batch_labels in load_data_in_batches('images/', 'dataset/train.csv', batch_size):
        images.extend(batch_images)
        labels.extend(batch_labels)

    # Convert lists to numpy arrays
    images = np.array(images)
    labels = np.array(labels)

    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)
    return X_train, X_val, y_train, y_val