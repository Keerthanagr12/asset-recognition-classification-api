import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from typing import Tuple, List

class ImagePreprocessor:
    """Image preprocessing and augmentation for Asset Recognition
    
    Handles:
    - Resizing to 224x224
    - Normalization with ImageNet statistics
    - Augmentation (rotation, flipping, brightness) for training
    - Tensor conversion
    """
    
    def __init__(self, img_size: int = 224):
        self.img_size = img_size
        
        # Normalization with ImageNet statistics
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        
        # Training augmentation pipeline
        self.train_transforms = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.RandomAffine(degrees=15, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            self.normalize
        ])
        
        # Validation/Test pipeline (no augmentation)
        self.val_transforms = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            self.normalize
        ])
    
    def preprocess_train(self, image: Image.Image) -> torch.Tensor:
        """Preprocess image for training (with augmentation)"""
        return self.train_transforms(image)
    
    def preprocess_val(self, image: Image.Image) -> torch.Tensor:
        """Preprocess image for validation/testing (no augmentation)"""
        return self.val_transforms(image)
    
    def preprocess_from_path(self, img_path: str, train: bool = False) -> torch.Tensor:
        """Load and preprocess image from file path"""
        image = Image.open(img_path).convert('RGB')
        if train:
            return self.preprocess_train(image)
        else:
            return self.preprocess_val(image)


def create_data_loaders(train_dir: str, val_dir: str, batch_size: int = 32) -> Tuple:
    """Create PyTorch data loaders for training and validation
    
    Args:
        train_dir: Path to training data directory
        val_dir: Path to validation data directory
        batch_size: Batch size for data loaders
    
    Returns:
        Tuple of (train_loader, val_loader)
    """
    preprocessor = ImagePreprocessor()
    
    train_dataset = transforms.ImageFolder(
        train_dir,
        transform=preprocessor.train_transforms
    )
    val_dataset = transforms.ImageFolder(
        val_dir,
        transform=preprocessor.val_transforms
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )
    
    return train_loader, val_loader, len(train_dataset.classes)
