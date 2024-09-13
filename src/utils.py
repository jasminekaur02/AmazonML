# src/utils.py
import requests
import os
import pandas as pd

def download_images_from_csv(csv_file, save_dir='images'):
    """
    Downloads images from URLs specified in a CSV file.

    Parameters:
    - csv_file: Path to the CSV file containing image links.
    - save_dir: Directory where images will be saved.
    """
    # Check if the CSV file exists
    if not os.path.exists(csv_file):
        print(f"Error: The file {csv_file} does not exist.")
        return

    # Create the directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"Created directory: {save_dir}")
    
    # Load the CSV file
    df = pd.read_csv(csv_file)
    
    # Check if 'image_link' column exists
    if 'image_link' not in df.columns:
        raise ValueError("CSV file must contain 'image_link' column.")
    
    # Download each image
    for link in df['image_link']:
        try:
            print(f"Downloading: {link}")  # Debugging statement
            response = requests.get(link)
            response.raise_for_status()  # Raise an error for bad responses
            image_name = os.path.join(save_dir, link.split('/')[-1])
            with open(image_name, 'wb') as f:
                f.write(response.content)
            print(f"Downloaded: {image_name}")
        except requests.exceptions.RequestException as e:
            print(f"Error downloading {link}: {e}")

# Example usage
if __name__ == "__main__":
    download_images_from_csv('dataset/train.csv')  # Download training images
    download_images_from_csv('dataset/test.csv')   # Download test images