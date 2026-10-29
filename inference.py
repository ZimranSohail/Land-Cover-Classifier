
from pathlib import Path
import json
import torch
from torch import nn
from torchvision import models
from PIL import Image
from torchvision import transforms

BASE_DIR = Path(__file__).resolve().parent

with open(BASE_DIR / "config.json") as f:
    CONFIG = json.load(f)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

def load_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(
        model.fc.in_features,
        CONFIG["num_classes"]
    )

    weights = torch.load(
        BASE_DIR / "model_weights.pth",
        map_location=DEVICE,
        weights_only=True
    )
    model.load_state_dict(weights)
    model.to(DEVICE)
    model.eval()
    return model

preprocess = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=CONFIG["normalization"]["mean"],
        std=CONFIG["normalization"]["std"]
    )
])

def predict(image, model=None):
    """Accept a PIL image and return predicted class and confidence."""
    if model is None:
        model = load_model()

    image = image.convert("RGB")
    tensor = preprocess(image).unsqueeze(0).to(DEVICE)

    with torch.inference_mode():
        probabilities = torch.softmax(model(tensor), dim=1)[0]
        index = int(probabilities.argmax())

    return {
        "class_id": index,
        "class_name": CONFIG["class_names"][index],
        "confidence": float(probabilities[index])
    }
