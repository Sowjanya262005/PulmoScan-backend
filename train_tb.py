# train_tb.py  (you can keep the old name if you want)
import os, json, torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet18, ResNet18_Weights
from tqdm import tqdm

# --------------------- Paths ---------------------
DATA_DIR = r"D:\data\tb"
SAVE_DIR = r"D:\lungcare-ai-starter-v2\backend\models"
os.makedirs(SAVE_DIR, exist_ok=True)

# --------------------- Transforms ----------------
IMG_SIZE = 224
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# --------------------- Build loaders -------------
def build_loaders():
    train_set = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=transform)
    val_set   = datasets.ImageFolder(os.path.join(DATA_DIR, "val"),   transform=transform)
    test_set  = datasets.ImageFolder(os.path.join(DATA_DIR, "test"),  transform=transform)

    # Save class order for inference
    with open(os.path.join(SAVE_DIR, "tb_classes.json"), "w") as f:
        json.dump(train_set.classes, f)
    print("Classes:", train_set.classes, train_set.class_to_idx)

    use_cuda = torch.cuda.is_available()
    # On GPU, a couple workers help; on CPU, workers=0 is simplest/stable
    num_workers = 2 if use_cuda else 0
    pin = use_cuda

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True,
                              num_workers=num_workers, pin_memory=pin,
                              persistent_workers=False)
    val_loader   = DataLoader(val_set, batch_size=32, shuffle=False,
                              num_workers=num_workers, pin_memory=pin,
                              persistent_workers=False)
    test_loader  = DataLoader(test_set, batch_size=32, shuffle=False,
                              num_workers=num_workers, pin_memory=pin,
                              persistent_workers=False)
    return train_loader, val_loader, test_loader, train_set.classes

# --------------------- Model ---------------------
def build_model(num_classes: int = 2) -> nn.Module:
    # Pretrained weights (downloads once and caches)
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

# --------------------- Train / Eval --------------
def train_one_epoch(model, loader, device, criterion, optimizer, epoch, epochs):
    model.train()
    running = 0.0
    pbar = tqdm(loader, desc=f"Epoch {epoch}/{epochs}", ncols=100)
    for x, y in pbar:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        running += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    return running / max(1, len(loader))

@torch.no_grad()
def eval_loader(model, loader, device, criterion):
    model.eval()
    correct = total = 0
    loss_sum = 0.0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        out = model(x)
        loss_sum += criterion(out, y).item()
        pred = out.argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    acc = correct / max(1, total)
    avg_loss = loss_sum / max(1, len(loader))
    return acc, avg_loss

# --------------------- Main ----------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using:", torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True  # speed up fixed-size inputs

    train_loader, val_loader, test_loader, classes = build_loaders()
    num_classes = len(classes)
    assert num_classes == 2, f"Expected 2 classes (NORMAL/TB), but found {num_classes}: {classes}"

    model = build_model(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    EPOCHS = 8
    best_acc = 0.0
    best_path = os.path.join(SAVE_DIR, "tb_model_best.pth")
    last_path = os.path.join(SAVE_DIR, "tb_model_last.pth")

    for epoch in range(1, EPOCHS + 1):
        tr_loss = train_one_epoch(model, train_loader, device, criterion, optimizer, epoch, EPOCHS)
        va_acc, va_loss = eval_loader(model, val_loader, device, criterion)
        print(f"train_loss={tr_loss:.4f}  val_loss={va_loss:.4f}  val_acc={va_acc*100:.2f}%")

        if va_acc > best_acc:
            best_acc = va_acc
            torch.save(model.state_dict(), best_path)
            print(f"✅ saved {os.path.basename(best_path)} (val_acc={va_acc*100:.2f}%)")

    te_acc, te_loss = eval_loader(model, test_loader, device, criterion)
    print(f"TEST acc={te_acc*100:.2f}%  loss={te_loss:.4f}")
    torch.save(model.state_dict(), last_path)
    print(f"💾 saved {os.path.basename(last_path)}")

if __name__ == "__main__":
    # Windows needs the main guard when DataLoader uses workers
    main()
