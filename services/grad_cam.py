# backend/services/grad_cam.py
import cv2
import numpy as np
import torch
from torch.nn import functional as F
from torchvision import transforms
from PIL import Image
import base64

IMG_SIZE = 224

def generate_gradcam(model, target_layer, img_pil: Image.Image, device, target_class: int, alpha=0.45):
    import cv2, numpy as np, torch, base64, io
    from torch.nn import functional as F
    from torchvision import transforms

    IMG_SIZE = 224
    model.eval()

    preprocess = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    img_tensor = preprocess(img_pil).unsqueeze(0).to(device)

    # Hook to grab gradients and activations
    gradients, activations = [], []
    def forward_hook(_, __, output): activations.append(output.detach())
    def backward_hook(_, grad_in, grad_out): gradients.append(grad_out[0].detach())
    fwd_handle = target_layer.register_forward_hook(forward_hook)
    bwd_handle = target_layer.register_full_backward_hook(backward_hook)

    # Forward + backward
    output = model(img_tensor)
    score = output[:, target_class]
    model.zero_grad()
    score.backward()

    fwd_handle.remove(); bwd_handle.remove()

    # Compute Grad-CAM
    grads, acts = gradients[0], activations[0]
    weights = grads.mean(dim=(2, 3), keepdim=True)
    cam = (weights * acts).sum(dim=1, keepdim=True)
    cam = F.relu(cam).squeeze().cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() + 1e-8)

    # Upscale + colormap
    cam_resized = cv2.resize(cam, (IMG_SIZE*2, IMG_SIZE*2))
    heatmap = np.uint8(255 * cam_resized)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_INFERNO)  # More dramatic colors

    # Add slight glow
    heatmap_blur = cv2.GaussianBlur(heatmap, (0,0), sigmaX=5)
    heatmap = cv2.addWeighted(heatmap, 0.7, heatmap_blur, 0.3, 0)

    # Overlay
    orig_img = np.array(img_pil.resize((IMG_SIZE*2, IMG_SIZE*2)))
    overlay = cv2.addWeighted(orig_img, 1 - alpha, heatmap, alpha, 0)

    # Add border + label
    overlay = cv2.copyMakeBorder(overlay, 10, 60, 10, 10, cv2.BORDER_CONSTANT, value=(0,0,0))
    label_text = f"Predicted: {target_class}"
    cv2.putText(overlay, label_text, (15, overlay.shape[0]-20),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)

    # Encode to base64
    def encode_img(img):
        _, buffer = cv2.imencode(".png", img)
        return base64.b64encode(buffer).decode("utf-8")

    return {
        "original": encode_img(orig_img),
        "overlay": encode_img(overlay),
        "heatmap": encode_img(heatmap)
    }
    