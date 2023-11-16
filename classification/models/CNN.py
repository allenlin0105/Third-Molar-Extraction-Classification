import math
from torch import nn

class CNN(nn.Module):
    def __init__(self, image_size): 
        super().__init__()
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=0)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.cnn2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=0)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.act = nn.ReLU()

        for _ in range(2):
            image_size -= 2
            image_size = math.floor(image_size / 2)
        
        self.fc = nn.Linear(32 * image_size * image_size, 3) 
    
    def forward(self, x):
        x = self.cnn1(x)
        x = self.act(x)
        x = self.maxpool1(x)

        x = self.cnn2(x)
        x = self.act(x)
        x = self.maxpool2(x)

        x = self.fc(x)
        return x