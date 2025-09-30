from fastapi import APIRouter, File, UploadFile, Query, HTTPException
from pathlib import Path
import torch
import numpy as np
from PIL import Image
import shutil
import tempfile

from backend.services.grad_cam import generate_gradcam

router = APIRouter()

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATHS = {
    "pneumonia": BASE_DIR / "pneumonia_model.pth",
    "tb": BASE_DIR / "models" / "tb_model_best.pth",
    "tuberculosis": BASE_DIR / "models" / "tb_model_best.pth",
    "lungcancer": BASE_DIR / "lung_cancer_model.pth",
    "lung cancer": BASE_DIR / "lung_cancer_model.pth",
}

CLASS_NAMES = {
    "pneumonia": ["NORMAL", "PNEUMONIA"],
    "tb": ["NORMAL", "TUBERCULOSIS"],
    "tuberculosis": ["NORMAL", "TUBERCULOSIS"],
    "lungcancer": ["ADENOCARCINOMA", "LARGE_CELL_CARCINOMA", "SQUAMOUS_CELL_CARCINOMA"],
    "lung cancer": ["ADENOCARCINOMA", "LARGE_CELL_CARCINOMA", "SQUAMOUS_CELL_CARCINOMA"],
}

gradcam_layers = {
    "pneumonia": lambda m: m.layer4[-1],
    "tb": lambda m: m.layer4[-1],
    "tuberculosis": lambda m: m.layer4[-1],
    "lungcancer": lambda m: m.layer4[-1],
    "lung cancer": lambda m: m.layer4[-1],
}

_model_cache = {}

def load_model(disease_key: str, device):
    from torchvision import models
    if disease_key in _model_cache:
        return _model_cache[disease_key]

    if disease_key.startswith("lung"):
        model = models.resnet50(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, len(CLASS_NAMES[disease_key]))
    else:
        model = models.resnet18(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, len(CLASS_NAMES[disease_key]))

    state = torch.load(MODEL_PATHS[disease_key], map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    _model_cache[disease_key] = model
    return model

from torchvision import transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

@router.post("/predict/{disease}")
async def predict_disease(
    disease: str,
    file: UploadFile = File(...),
    explain: int = Query(0)
):
    disease_key = disease.lower()
    if disease_key not in MODEL_PATHS:
        raise HTTPException(status_code=400, detail="Unknown disease type")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(disease_key, device)

    # Save uploaded image to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    # Predict
    img = Image.open(tmp_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)[0].cpu().numpy()
        pred_idx = int(np.argmax(probs))
        label = CLASS_NAMES[disease_key][pred_idx]
        confidence = float(probs[pred_idx])

    # Grad-CAM
    explain_image = None
    explain_supported = False
    if explain == 1:
        from backend.services.grad_cam import generate_gradcam  # Import only when needed
        try:
            target_layer = gradcam_layers[disease_key](model)
            explain_image = generate_gradcam(model, target_layer, tmp_path, device)
            explain_supported = True
        except Exception as e:
            print("Grad-CAM failed:", e)
            explain_image = None
            explain_supported = False


    return {
        "label": label,
        "confidence": confidence,
        "probs": probs.tolist(),
        "classes": CLASS_NAMES[disease_key],
        "filename": Path(file.filename).name,
        "explain_requested": bool(explain),
        "explain_supported": explain_supported,
        "explain_image": explain_image
    }
