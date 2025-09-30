import torch, timm
from PIL import Image
from .transforms import infer_tf
from ..config import IMG_SIZE, EXPLAIN
from .grad_cam import GradCAM, overlay_heatmap

class Classifier:
    def __init__(self, weights_path, class_names, backbone='resnet18'):
        self.model = timm.create_model(backbone, pretrained=False, num_classes=len(class_names))
        state = torch.load(weights_path, map_location='cpu')
        self.model.load_state_dict(state)
        self.model.eval()
        self.class_names = class_names
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)

        # Grad-CAM target layer guess (works for resnet-like)
        self.target_layer = None
        for n,m in self.model.named_modules():
            if "layer4" in n and hasattr(m, "conv2") or hasattr(m, "bn2"):
                self.target_layer = m
        if self.target_layer is None:
            # fallback to last layer
            self.target_layer = list(self.model.modules())[-2]

    @torch.inference_mode()
    def predict(self, img: Image.Image, explain: bool=False):
        tf = infer_tf(IMG_SIZE)
        x = tf(img).unsqueeze(0).to(self.device)

        if explain or EXPLAIN:
            # Turn off inference mode for backprop
            with torch.enable_grad():
                cammer = GradCAM(self.model, self.target_layer)
                cam, logits = cammer(x, target_index=None)  # target later
                probs = torch.softmax(logits, dim=1)[0]
                score, idx = probs.max(dim=0)
                # re-run for chosen index to get tailored cam
                self.model.zero_grad(set_to_none=True)
                cam, logits = cammer(x, target_index=idx.item())
                heatmap_b64 = overlay_heatmap(img, cam)
        else:
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)[0]
            score, idx = probs.max(dim=0)
            heatmap_b64 = None

        topk = torch.topk(probs, k=min(3, len(self.class_names)))
        return {
            'label': self.class_names[idx.item()],
            'score': float(score.item()),
            'topk_labels': [self.class_names[i] for i in topk.indices.tolist()],
            'topk_scores': [float(s) for s in topk.values.tolist()],
            'heatmap_b64': heatmap_b64,
        }
