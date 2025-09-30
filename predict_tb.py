# backend/routers/predict_tb.py
from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from torchvision.models import resnet18, ResNet18_Weights
from torchvision import transforms
from torch import nn
from PIL import Image
from pathlib import Path
import torch, io

from backend.services.grad_cam import generate_gradcam

router = APIRouter(prefix="/predict", tags=["TB"])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = Path(r"D:\lungcare-ai-starter-v2\backend\models\tb_model_best.pth")
CLASSES = ["NORMAL", "TB"]

# Build & load model
_model = resnet18(weights=ResNet18_Weights.DEFAULT)
_model.fc = nn.Linear(_model.fc.in_features, len(CLASSES))

if not MODEL_PATH.exists():
    raise RuntimeError(f"TB model not found at: {MODEL_PATH}")
state = torch.load(MODEL_PATH, map_location=device)
_model.load_state_dict(state, strict=True)
_model.to(device).eval()

_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

def _preprocess_image(file_bytes: bytes) -> torch.Tensor:
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    return _transform(img).unsqueeze(0)

@router.post("/tb")
async def predict_tb(file: UploadFile = File(...), explain: int = Query(0)):
    try:
        if not file or not file.filename:
            raise HTTPException(status_code=400, detail="No file uploaded.")
        content = await file.read()

        x = _preprocess_image(content).to(device)
        with torch.no_grad():
            logits = _model(x)
            probs = torch.softmax(logits, dim=1)[0]
            idx = int(torch.argmax(probs).item())
            conf = float(probs[idx].item())

        explain_image_b64 = None
        if explain == 1:
            explain_image_b64 = generate_gradcam(_model, x, target_class=idx)

        return {
            "label": CLASSES[idx],
            "confidence": round(conf, 6),
            "probs": [float(p) for p in probs.cpu().numpy().tolist()],
            "classes": CLASSES,
            "filename": file.filename,
            "explain_requested": bool(explain),
            "explain_supported": True,
            "explain_image": explain_image_b64,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
