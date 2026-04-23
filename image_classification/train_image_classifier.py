import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np
import os

# --- 1. Configuration ---
DATA_DIR = 'C:/Users/Hafizi/Documents/computer vision/03-lens/data/lens.v67_clip'
BATCH_SIZE = 16
EPOCHS = 100
LR = 0.0001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_SAVE_PATH = "best_model.pth"

# --- 2. Data Preparation & Dual Transforms ---

# Helper class to apply different transforms to training and validation splits
class ApplyTransform(torch.utils.data.Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
        
    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y
        
    def __len__(self):
        return len(self.subset)

# A. Training Augmentations (Hue, Brightness, Rotation, Flips)
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(
        brightness=0.3, 
        contrast=0.3, 
        saturation=0.2, 
        hue=0.1
    ),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# B. Validation Transforms (Clean - No Augmentation)
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load dataset and split
full_dataset = datasets.ImageFolder(DATA_DIR)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
base_train, base_val = random_split(full_dataset, [train_size, val_size])

# Wrap splits with their respective transforms
train_data = ApplyTransform(base_train, transform=train_transform)
val_data = ApplyTransform(base_val, transform=val_transform)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)

class_names = full_dataset.classes
print(f"Classes found: {class_names} | Total images: {len(full_dataset)}")
print(f"Training set: {len(train_data)} | Validation set: {len(val_data)}")

# --- 3. Model Definition (MobileNetV2) ---
# model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
# num_ftrs = model.classifier[1].in_features
# model.classifier[1] = nn.Linear(num_ftrs, 2)

model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1)
num_ftrs = model.classifier[3].in_features # Note: Index is 3 for V3
model.classifier[3] = nn.Linear(num_ftrs, 2)

# Freeze features to focus training on the classifier head
for param in model.features.parameters():
    param.requires_grad = False


model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
# mobilenet_v2
# optimizer = optim.Adam(model.classifier[1].parameters(), lr=LR)

# mobilenet_v3
optimizer = optim.Adam(model.classifier[3].parameters(), lr=LR)

# --- 4. Training and Evaluation Loop ---
def evaluate():
    model.eval()
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            
    return y_true, y_pred

print("\nStarting Training with Augmentations...")
best_acc = 0.0

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
    
    epoch_loss = running_loss / len(train_data)
    y_true, y_pred = evaluate()
    acc = accuracy_score(y_true, y_pred)
    
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {epoch_loss:.4f} | Val Acc: {acc:.4f}")
    
    # Save the best model based on validation accuracy
    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), MODEL_SAVE_PATH)

# --- 5. Final Detailed Metrics ---
print("\n" + "="*30)
print("FINAL EVALUATION METRICS (Best Model)")
print("="*30)

# Load best weights for final report
model.load_state_dict(torch.load(MODEL_SAVE_PATH))
y_true, y_pred = evaluate()

print(classification_report(y_true, y_pred, target_names=class_names))
print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))

print(f"\nTraining complete. Best model saved to {MODEL_SAVE_PATH}")


# Inference
# import torch
# import torch.nn as nn
# from torchvision import models, transforms
# from PIL import Image

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# MODEL_PATH = "best_model.pth"
# CLASS_NAMES = ['class_0', 'class_1'] 

# model = models.mobilenet_v3_large(weights=None)
# num_ftrs = model.classifier[3].in_features
# model.classifier[3] = nn.Linear(num_ftrs, len(CLASS_NAMES))

# model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
# model.to(DEVICE)
# model.eval()

# inference_transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# ])

# def predict_image(image_path):
#     img = Image.open(image_path).convert('RGB')
    
#     img_tensor = inference_transform(img).unsqueeze(0).to(DEVICE)
    
#     with torch.no_grad():
#         outputs = model(img_tensor)
#         probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
#         confidence, predicted_idx = torch.max(probabilities, 0)
    
#     label = CLASS_NAMES[predicted_idx.item()]
#     print(f"Prediction: {label} ({confidence.item()*100:.2f}%)")
#     return label

# Example usage:
# predict_image('path_to_your_test_image.jpg')