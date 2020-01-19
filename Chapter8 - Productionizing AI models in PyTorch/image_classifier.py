import io
import torch
import torch.nn as nn
from torchvision import models
from PIL import Image
import torchvision.transforms as transforms

def create_model():

    model_path = "checkpoint_final.pth"
    model = models.densenet161(pretrained=True)
    model.classifier = nn.Linear(2208, 102)
    model.load_state_dict(torch.load(model_path, map_location='cpu'), strict=False)
    model.eval()
    return model

def image_transformer():
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])