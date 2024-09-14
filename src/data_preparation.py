import pandas as pd
import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split

def load_data(image_dir, csv_file):
    """
    Load images and labels from the specified CSV file.

    Parameters:
    - image_dir: Directory where images are stored.
    - csv_file: Path to the CSV file containing image links and labels.

    Returns:
    - images: Numpy array of loaded images.
    - labels: Numpy array of corresponding labels.
    """
    df = pd.read_csv(csv_file)
    images = []
    labels = []
    
    for index, row in df.iterrows():
        img_path = os.path.join(image_dir, row['image_link'].split('/')[-1])
        image = cv2.imread(img_path)
        
        if image is None:
            continue  # Skip this image if it cannot be loaded
        
        image = cv2.resize(image, (128, 128))  # Resize to fit model input
        images.append(image)
        
        # Convert the label to a numeric type
        label = row['entity_value']
        try:
            labels.append(float(label))  # Convert to float (or int if applicable)
        except ValueError:
            print(f"Warning: Unable to convert label '{label}' to float. Skipping this entry.")
            continue

    return np.array(images), np.array(labels)