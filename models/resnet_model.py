from torch import nn
from torchvision import models  

class ResNetAudio(nn.Module):
    def __init__(self, backbone,num_classes):
        super(ResNetAudio, self).__init__()
        self.resnet = backbone  # Load ResNet18
        num_features = self.resnet.fc.in_features
        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.resnet.fc = nn.Sequential(
                nn.Linear(num_features, 256),  # 512 → 256
                nn.ReLU(),
                nn.Dropout(0.5),  # Dropout to prevent overfitting
                nn.Linear(256, 128),  # 256 → 128
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(128, num_classes)  # 128 → Output classes
            )
    def forward(self, x):
        return self.resnet(x)