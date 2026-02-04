import torch
import torch.nn as nn
from typing import Tuple

class AssetCNN(nn.Module):
    """Convolutional Neural Network for Asset Recognition & Classification
    
    Architecture:
    - 3 Conv2D layers (32, 64, 128 filters) with 3x3 kernels
    - MaxPooling after each conv layer
    - Dropout for regularization (0.25)
    - Fully connected layers for classification
    
    Args:
        num_classes (int): Number of output classes. Default: 8 (e-commerce categories)
    """
    
    def __init__(self, num_classes: int = 8):
        super(AssetCNN, self).__init__()
        
        # Conv Block 1: 3 -> 32 channels
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.25)
        
        # Conv Block 2: 32 -> 64 channels
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout(0.25)
        
        # Conv Block 3: 64 -> 128 channels
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.dropout3 = nn.Dropout(0.25)
        
        # Global Average Pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # FC layers
        self.fc1 = nn.Linear(128, 256)
        self.relu_fc1 = nn.ReLU(inplace=True)
        self.dropout_fc = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network
        
        Args:
            x: Input tensor of shape (batch_size, 3, 224, 224)
        
        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        # Conv Block 1
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.dropout1(x)
        
        # Conv Block 2
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.dropout2(x)
        
        # Conv Block 3
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        x = self.dropout3(x)
        
        # Global average pooling
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        
        # FC layers
        x = self.fc1(x)
        x = self.relu_fc1(x)
        x = self.dropout_fc(x)
        x = self.fc2(x)
        
        return x


def create_model(num_classes: int = 8, device: str = 'cpu') -> AssetCNN:
    """Factory function to create and initialize AssetCNN model
    
    Args:
        num_classes: Number of output classes
        device: Device to create model on ('cpu' or 'cuda')
    
    Returns:
        Initialized AssetCNN model on specified device
    """
    model = AssetCNN(num_classes=num_classes)
    model = model.to(device)
    return model
