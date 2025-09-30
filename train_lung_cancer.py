# train_lung_cancer.py
import os, torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tqdm import tqdm

DATA_DIR = r"D:\data\lungcancer"  # adjust
BATCH = 16
EPOCHS = 10
IMG_SIZE = 224
LR = 1e-3
NUM_CLASSES = 3

def get_loaders():
    tfm = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    train_ds = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), tfm)
    val_ds   = datasets.ImageFolder(os.path.join(DATA_DIR, "val"), tfm)

    use_cuda = torch.cuda.is_available()
    num_workers = (os.cpu_count() // 2) if use_cuda else 0
    pin = use_cuda

    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True,
                              num_workers=num_workers, pin_memory=pin,
                              persistent_workers=False)
    val_loader = DataLoader(val_ds, batch_size=BATCH, shuffle=False,
                            num_workers=num_workers, pin_memory=pin,
                            persistent_workers=False)
    return train_loader, val_loader

def build_model():
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model

def train_one_epoch(model, loader, device, criterion, optimizer, epoch):
    model.train()
    running = 0.0
    for x, y in tqdm(loader, desc=f"Epoch {epoch}/{EPOCHS}"):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        running += loss.item() * x.size(0)
    return running / len(loader.dataset)

@torch.no_grad()
def validate(model, loader, device, criterion):
    model.eval()
    correct, total, running = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss = criterion(out, y)
        running += loss.item() * x.size(0)
        pred = out.argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return running / len(loader.dataset), correct / total

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using:", "GPU" if device.type == "cuda" else "CPU")

    train_loader, val_loader = get_loaders()
    model = build_model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR)

    for epoch in range(1, EPOCHS + 1):
        tr_loss = train_one_epoch(model, train_loader, device, criterion, optimizer, epoch)
        val_loss, val_acc = validate(model, val_loader, device, criterion)
        print(f"Epoch {epoch}/{EPOCHS} | Train Loss: {tr_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%")

    torch.save(model.state_dict(), "lung_cancer_model.pth")
    print("✅ Saved: lung_cancer_model.pth")

if __name__ == "__main__":
    # For Windows safety with multiprocessing (even if we set workers=0 on CPU)
    # import torch.multiprocessing as mp
    # mp.set_start_method("spawn", force=True)
    main()
