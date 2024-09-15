import os
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torchvision import models
from torchvision import transforms
from PIL import Image

# Preprocess the image for ResNet
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def extract_features_from_image(model, image_path):
    try:
        img = Image.open(image_path)
        img_tensor = preprocess(img).unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            features = model(img_tensor).squeeze().numpy()
        return features
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

def prepare_data(csv_file, image_dir='images', model=None, limit=100):
    df = pd.read_csv(csv_file)
    features = []
    entity_data = []

    for i, row in df.iterrows():
        if i >= limit:
            break
        image_path = os.path.join(image_dir, row['image_link'].split('/')[-1])
        image_features = extract_features_from_image(model, image_path)
        if image_features is not None:
            features.append(image_features)
            entity_info = {
                'entity_name': row['entity_name'],
                'entity_value': row['entity_value']
            }
            entity_data.append(entity_info)
    
    return features, entity_data

class EntityPredictorModel(torch.nn.Module):
    def __init__(self, input_size):
        super(EntityPredictorModel, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, 128)
        self.fc2 = torch.nn.Linear(128, 64)
        self.fc3 = torch.nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def load_model(model_path, input_size):
    model = EntityPredictorModel(input_size)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def train_model(X_train, y_train, X_val, y_val, input_size, epochs=20, batch_size=32):
    model = EntityPredictorModel(input_size)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val)

        print(f"Epoch {epoch+1}/{epochs}, Training Loss: {loss.item()}, Validation Loss: {val_loss.item()}")

    torch.save(model.state_dict(), 'models/entity_predictor.pth')
    print("Model saved as 'entity_predictor.pth'.")

def predict(model, input_data):
    with torch.no_grad():
        input_tensor = torch.tensor(input_data, dtype=torch.float32)
        predictions = model(input_tensor)
    return predictions.numpy()

if __name__ == "__main__":
    resnet_model = models.resnet50(weights='IMAGENET1K_V1')
    resnet_model = torch.nn.Sequential(*list(resnet_model.children())[:-1])

    csv_file = 'dataset/train.csv'
    features, entity_data = prepare_data(csv_file, model=resnet_model, limit=100)

    features = np.array(features)

    # Convert entity values to numeric using pandas
    entity_values = pd.to_numeric([data['entity_value'] for data in entity_data], errors='coerce')
    entity_values = entity_values.dropna().to_numpy()  # Drop NaN values and convert to NumPy array

    # Ensure features match the length of entity_values
    features = features[:len(entity_values)]

    X_train, X_val, y_train, y_val = train_test_split(features, entity_values, test_size=0.2, random_state=42)

    input_size = X_train.shape[1]
    train_model(X_train, y_train, X_val, y_val, input_size, epochs=20, batch_size=32)

    model_path = 'models/entity_predictor.pth'
    model = load_model(model_path, input_size)

    new_images_csv = 'dataset/new_images.csv'
    new_df = pd.read_csv(new_images_csv)

    new_features = []
    image_dir = 'images'

    for index, row in new_df.iterrows():
        image_path = os.path.join(image_dir, row['image_link'].split('/')[-1])
        image_features = extract_features_from_image(resnet_model, image_path)
        if image_features is not None:
            new_features.append(image_features)

    new_features = np.array(new_features)

    predictions = predict(model, new_features)
    print("Predictions:", predictions)