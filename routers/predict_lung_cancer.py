# backend/routers/predict_lung_cancer.py
import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from fastapi import APIRouter, File, UploadFile, HTTPException, Query
from PIL import Image
from io import BytesIO

from backend.services.grad_cam import generate_gradcam

router = APIRouter(prefix="/predict", tags=["Lung Cancer"])

# Adjust to your actual file location if different:
MODEL_PATH = r"D:\lungcare-ai-starter-v2\backend\lung_cancer_model.pth"

# ⚠ IMPORTANT: This must match the class order used during training.
# If you saved train_ds.classes to a JSON, load it instead of hardcoding.
CLASS_NAMES = ["adenocarcinoma", "squamous cell carcinoma", "normal"]
NUM_CLASSES = len(CLASS_NAMES)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model (ResNet50 as in training)
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"Lung cancer model not found at: {MODEL_PATH}")

state = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(state, strict=True)
model.to(device).eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

@router.post("/lungcancer")
async def predict_lung_cancer(file: UploadFile = File(...), explain: int = Query(0)):
    try:
        if not file or not file.filename:
            raise HTTPException(status_code=400, detail="No file uploaded.")

        image_bytes = await file.read()
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        img_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(img_tensor)
            probs_t = torch.softmax(outputs, dim=1)[0]
            pred_idx = int(torch.argmax(probs_t).item())
            confidence = float(probs_t[pred_idx].item())
            probs = [float(p) for p in probs_t.cpu().numpy().tolist()]

        explain_image_b64 = None
        if explain == 1:
            explain_image_b64 = generate_gradcam(model, img_tensor, target_class=pred_idx)

        return {
            "label": CLASS_NAMES[pred_idx],
            "confidence": confidence,
            "probs": probs,
            "classes": CLASS_NAMES,
            "filename": file.filename,
            "explain_requested": bool(explain),
            "explain_supported": True,
            "explain_image": explain_image_b64
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
