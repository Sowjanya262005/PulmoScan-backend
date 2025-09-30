import os
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torchvision.models import resnet18, ResNet18_Weights
from tqdm import tqdm  # Optional: pip install tqdm

# Paths
data_dir = r"D:\data\pneumonia"
train_dir = os.path.join(data_dir, "train")
val_dir = os.path.join(data_dir, "val")
test_dir = os.path.join(data_dir, "test")

# Image transformations (with proper normalization)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# Datasets and loaders
train_set = datasets.ImageFolder(train_dir, transform=transform)
val_set = datasets.ImageFolder(val_dir, transform=transform)
test_set = datasets.ImageFolder(test_dir, transform=transform)

train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
val_loader = DataLoader(val_set, batch_size=32)
test_loader = DataLoader(test_set, batch_size=32)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model
weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights)
model.fc = nn.Linear(model.fc.in_features, 2)  # NORMAL vs PNEUMONIA
model = model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training + Validation Loop
epochs = 5
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    print(f"\n📦 Epoch {epoch+1}/{epochs}...")

    for images, labels in tqdm(train_loader, desc="🔁 Training"):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    print(f"🔹 Training Loss: {avg_loss:.4f}")

    # Validation phase
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"✅ Validation Accuracy: {accuracy:.2f}%")

# Save model
torch.save(model.state_dict(), "pneumonia_model.pth")
print("💾 Model saved as 'pneumonia_model.pth'")
