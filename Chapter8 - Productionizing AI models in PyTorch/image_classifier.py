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
