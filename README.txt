Satellite Land-Cover Classification Model

Architecture: ResNet18, fine-tuned from ImageNet weights
Input: RGB image, 128x128 pixels
Classes: See config.json

Files:
- model_weights.pth: trained model parameters
- inference.py: model loading and prediction functions
- config.json: class mapping and preprocessing settings
- requirements.txt: Python dependencies

Example:
from PIL import Image
from inference import load_model, predict

model = load_model()
result = predict(Image.open('sample.png'), model)
print(result)

Important: inference images should use compatible Sentinel-2 RGB
imagery and the same visual scaling used by the training dataset.
