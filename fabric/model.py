from pathlib import Path
import torch
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights
from torchvision import transforms
from PIL import Image, ImageOps

MEAN = [.485, .456, .406]
STD = [.229, .224, .225]

def transform(size=128):
    return transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor(), transforms.Normalize(MEAN, STD)])

def build_model(pretrained=False):
    model = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model

def open_image(source):
    with Image.open(source) as image:
        if image.width * image.height > 25_000_000:
            raise ValueError("Image exceeds 25 megapixels. Resize it first.")
        return ImageOps.exif_transpose(image).convert("RGB")

def load_checkpoint(path):
    # Never unpickle arbitrary uploaded Python objects.
    saved = torch.load(Path(path), map_location="cpu", weights_only=True)
    if saved.get("format_version") != 1 or saved.get("classes") != ["defect", "normal"]:
        raise ValueError("Unsupported checkpoint format or class mapping")
    model = build_model()
    model.load_state_dict(saved["state_dict"])
    model.eval()
    return model, saved

class GradCAM:
    """Class activation map from the last residual block; not segmentation."""
    def __init__(self, model):
        self.model = model
        self.activation = None
        self.gradient = None
        self.handle = model.layer4.register_forward_hook(self._capture)

    def _capture(self, module, inputs, output):
        self.activation = output
        if output.requires_grad:
            output.register_hook(self._gradient)

    def _gradient(self, grad):
        self.gradient = grad

    def __call__(self, tensor, target=None):
        self.model.eval()
        self.model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            logits = self.model(tensor)
            index = int(logits.argmax(1).item()) if target is None else int(target)
            if index not in (0, 1):
                raise ValueError("target must be 0 or 1")
            logits[0, index].backward()
            weights = self.gradient.mean((2, 3), keepdim=True)
            cam = (weights * self.activation).sum(1, keepdim=True).relu()
            cam = torch.nn.functional.interpolate(cam, size=tensor.shape[-2:], mode="bilinear", align_corners=False)
            cam = cam[0, 0].detach()
            maximum = cam.max()
            if maximum > 0:
                cam = cam / maximum
        return logits.detach().softmax(1)[0], cam.numpy(), index

    def close(self):
        self.handle.remove()

def overlay(image, cam, alpha=.5):
    import numpy as np
    heat = Image.fromarray((np.clip(cam, 0, 1)*255).astype("uint8")).resize(image.size)
    v = np.asarray(heat, dtype=float)/255
    color = np.stack([255*v, 160*np.sqrt(v), 255*(1-v)*.3], axis=-1).astype("uint8")
    # Only high activations tint the image strongly.
    blend = alpha*v[..., None]
    return Image.fromarray(np.clip(np.asarray(image)*(1-blend) + color*blend, 0, 255).astype("uint8"))
