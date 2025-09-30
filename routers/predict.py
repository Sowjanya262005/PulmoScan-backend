from typing import List
import io
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from fastapi import APIRouter, UploadFile, File, HTTPException, Query

from backend.services.grad_cam import generate_gradcam

# ---------------- Router ----------------
router = APIRouter(prefix="/predict", tags=["PulmoScan Predict"])

# ---------------- Config ----------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 224

# Model paths
PNEUMONIA_MODEL_PATH = Path(r"D:\lungcare-ai-starter-v2\backend\pneumonia_model.pth")
TB_MODEL_PATH        = Path(r"D:\lungcare-ai-starter-v2\backend\models\tb_model_best.pth")
LUNG_MODEL_PATH      = Path(r"D:\lungcare-ai-starter-v2\backend\lung_cancer_model.pth")

# Class names
PNEUMONIA_CLASSES: List[str] = ["NORMAL", "PNEUMONIA"]
TB_CLASSES:        List[str] = ["NORMAL", "TB"]
LUNG_CLASSES:      List[str] = ["adenocarcinoma", "normal", "squamous cell carcinoma"]

# ---------------- Preprocessing ----------------
tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

def _read_image(b: bytes) -> Image.Image:
    return Image.open(io.BytesIO(b)).convert("RGB")

def _prep(img: Image.Image) -> torch.Tensor:
    return tfms(img).unsqueeze(0).to(DEVICE)

# ---------------- Model loading ----------------
def _load_generic(path: Path, num_classes: int, arch: str = "resnet18") -> nn.Module:
    if not path.exists():
        raise FileNotFoundError(str(path))

    # Build architecture based on provided arch
    if arch == "resnet18":
        model = models.resnet18(weights=None)
    elif arch == "resnet50":
        model = models.resnet50(weights=None)
    else:
        raise ValueError(f"Unsupported architecture: {arch}")

    # Adjust classifier head
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    # Load weights
    obj = torch.load(path, map_location=DEVICE)
    if isinstance(obj, dict):
        state = obj.get("state_dict", obj)
        state = {k.replace("module.", ""): v for k, v in state.items()}
        model.load_state_dict(state, strict=False)
    else:
        model = obj

    model.to(DEVICE).eval()
    return model

# Load models with correct architectures
try:
    PNEUMONIA_MODEL = _load_generic(PNEUMONIA_MODEL_PATH, len(PNEUMONIA_CLASSES), arch="resnet18")
except Exception as e:
    print("⚠️ Failed to load pneumonia model:", e)
    PNEUMONIA_MODEL = None

try:
    TB_MODEL = _load_generic(TB_MODEL_PATH, len(TB_CLASSES), arch="resnet18")
except Exception as e:
    print("⚠️ Failed to load TB model:", e)
    TB_MODEL = None

try:
    LUNG_MODEL = _load_generic(LUNG_MODEL_PATH, len(LUNG_CLASSES), arch="resnet50")
except Exception as e:
    print("⚠️ Failed to load lung cancer model:", e)
    LUNG_MODEL = None

# ---------------- Prediction logic ----------------
def _predict(model: nn.Module, classes: List[str], img: Image.Image, explain: bool) -> dict:
    if model is None:
        raise HTTPException(status_code=503, detail="Model not available on server.")

    x = _prep(img)
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1).cpu().numpy().ravel().tolist()
        idx = int(logits.argmax(1).item())

    response = {
        "label": classes[idx],
        "confidence": float(probs[idx]),
        "probs": [float(p) for p in probs],
        "classes": classes
    }

    if explain:
        target_layer = model.layer4[-1]
        gradcam_images = generate_gradcam(model, target_layer, img, DEVICE, idx)

        # New fields
        response["explain_original"] = gradcam_images["original"]
        response["explain_overlay"] = gradcam_images["overlay"]
        response["explain_heatmap"] = gradcam_images["heatmap"]

        # Backward compatibility
        response["explain_image"] = gradcam_images["overlay"]
    else:
        response["explain_original"] = None
        response["explain_overlay"] = None
        response["explain_heatmap"] = None
        response["explain_image"] = None

    return response

# ---------------- Endpoints ----------------
@router.get("/health")
def health():
    return {
        "device": str(DEVICE),
        "pneumonia_model": str(PNEUMONIA_MODEL_PATH),
        "tb_model": str(TB_MODEL_PATH),
        "lung_model": str(LUNG_MODEL_PATH),
        "pneumonia_loaded": PNEUMONIA_MODEL is not None,
        "tb_loaded": TB_MODEL is not None,
        "lung_loaded": LUNG_MODEL is not None
    }

@router.post("/pneumonia")
async def predict_pneumonia(
    file: UploadFile = File(...),
    explain: int = Query(0, description="1 = return Grad-CAM visualizations")
):
    if not file or not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded.")
    img = _read_image(await file.read())
    return _predict(PNEUMONIA_MODEL, PNEUMONIA_CLASSES, img, explain == 1)

@router.post("/tuberculosis")
async def predict_tuberculosis(
    file: UploadFile = File(...),
    explain: int = Query(0, description="1 = return Grad-CAM visualizations")
):
    if not file or not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded.")
    img = _read_image(await file.read())
    return _predict(TB_MODEL, TB_CLASSES, img, explain == 1)

@router.post("/lung_cancer")
async def predict_lung_cancer(
    file: UploadFile = File(...),
    explain: int = Query(0, description="1 = return Grad-CAM visualizations")
):
    if not file or not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded.")
    img = _read_image(await file.read())
    return _predict(LUNG_MODEL, LUNG_CLASSES, img, explain == 1)

# Optional alias for space in URL
@router.post("/lung cancer")
async def predict_lung_cancer_alias(
    file: UploadFile = File(...),
    explain: int = Query(0)
):
    return await predict_lung_cancer(file, explain)
