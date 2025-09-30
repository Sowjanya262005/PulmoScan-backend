from fastapi import APIRouter, UploadFile, File, Query, HTTPException
import torch
import shutil
import os
import base64
from torchvision import transforms, models
from PIL import Image
from backend.services.grad_cam import generate_gradcam  # ✅ correct path

router = APIRouter()

# Paths
MODEL_PATH = r"D:\lungcare-ai-starter-v2\backend\pneumonia_model.pth"
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Classes for Pneumonia
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, len(CLASS_NAMES))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval().to(device)

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

@router.post("/predict/pneumonia")
async def predict_pneumonia(
    file: UploadFile = File(...),
    explain: int = Query(0, description="Set 1 to generate Grad-CAM heatmap")
):
    try:
        # Save uploaded file temporarily
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Load and preprocess image
        image = Image.open(file_path).convert("RGB")
        input_tensor = transform(image).unsqueeze(0).to(device)

        # Predict
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.nn.functional.softmax(outputs[0], dim=0)
            confidence, predicted_class = torch.max(probs, 0)

        # Grad-CAM
        explain_image_b64 = None
        explain_supported = False
        if explain == 1:
            try:
                heatmap_path = os.path.join(UPLOAD_DIR, f"cam_{file.filename}")
                generate_gradcam(
                    model=model,
                    image_path=file_path,
                    target_layer="layer4",  # last conv layer for ResNet18
                    output_path=heatmap_path,
                    device=device
                )
                with open(heatmap_path, "rb") as img_file:
                    explain_image_b64 = base64.b64encode(img_file.read()).decode("utf-8")
                explain_supported = True
            except Exception as e:
                print(f"Grad-CAM failed: {e}")

        return {
            "label": CLASS_NAMES[predicted_class.item()],
            "confidence": confidence.item(),
            "probs": probs.tolist(),
            "classes": CLASS_NAMES,
            "filename": file.filename,
            "explain_requested": bool(explain),
            "explain_supported": explain_supported,
            "explain_image": explain_image_b64
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
