import os
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd

# Preprocess the image for ResNet (example)
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def extract_features_from_image(model, image_path):
    """
    Extracts features from an image using a pre-trained model.

    Args:
    - model: Pre-trained PyTorch model.
    - image_path: Path to the image.

    Returns:
    - features: Extracted features as a numpy array.
    """
    try:
        img = Image.open(image_path)
        img_tensor = preprocess(img).unsqueeze(0)  # Add batch dimension

        # Extract features
        with torch.no_grad():
            features = model(img_tensor).squeeze().numpy()
        
        return features
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

def prepare_data(csv_file, image_dir='images', model=None, limit=100):
    """
    Prepares data and extracts features using a pre-trained model.

    Args:
    - csv_file: Path to the CSV file.
    - image_dir: Directory where images are stored.
    - model: Pre-trained PyTorch model for feature extraction.
    - limit: The number of images to process.

    Returns:
    - features: Extracted features for each image.
    - entity_data: Corresponding entity information.
    """
    df = pd.read_csv(csv_file)
    features = []
    entity_data = []

    # Process only the first `limit` images
    for i, row in df.iterrows():
        if i >= limit:
            break

        image_path = os.path.join(image_dir, row['image_link'].split('/')[-1])
        
        # Extract features using the pre-trained model
        image_features = extract_features_from_image(model, image_path)
        if image_features is not None:
            features.append(image_features)

            # Store the entity data
            entity_info = {
                'entity_name': row['entity_name'],
                'entity_value': row['entity_value']
            }
            entity_data.append(entity_info)
    
    return features, entity_data

if __name__ == "__main__":
    # Load pre-trained ResNet model
    resnet_model = models.resnet50(pretrained=True)
    resnet_model = torch.nn.Sequential(*list(resnet_model.children())[:-1])  # Remove classification layer

    # Prepare data (limit to 100 images)
    csv_file = 'dataset/train.csv'
    features, entity_data = prepare_data(csv_file, model=resnet_model, limit=100)
    print(f"Extracted {len(features)} feature vectors.")
