import torch
from torch import nn
from torchvision import models, transforms

class GoogLeNet(nn.Module):
    def __init__(self): 
        super().__init__()
        self.normalize =  transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        self.resnet = models.googlenet(pretrained=True, progress=True)
        self.fc = nn.Linear(1000, 3) 
    
    def forward(self, x: torch.Tensor):
        """
        x: a batch of images with shape (batch_size, 3, image_size, image_size)
        return: a batch of predicted probabilities with shape (batch_size, 3)
        """
        x = x.float()
        x = self.normalize(x)
        x = self.resnet(x)
        x = self.fc(x)
        return x