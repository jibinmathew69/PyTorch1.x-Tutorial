import io
import torch
import torch.nn as nn
from torchvision import models
from PIL import Image
import torchvision.transforms as transforms

def create_model():

    model_path = "image_model.pth"
    model = models.densenet161(pretrained=True)