"""
Demonstration of computer vision tasks using PyTorch and torchvision.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
import cv2
from PIL import Image

@dataclass
class CVConfig:
    """Configuration for computer vision tasks."""
    batch_size: int = 32
    learning_rate: float = 0.001
    num_epochs: int = 10
    image_size: int = 224
    num_classes: int = 10
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

class ConvNet(nn.Module):
    """Custom CNN architecture."""
    
    def __init__(self, num_classes: int):
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def prepare_data(config: CVConfig) -> Tuple[DataLoader, DataLoader]:
    """Prepare CIFAR-10 dataset."""
    
    transform = transforms.Compose([
        transforms.Resize(config.image_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )
    
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=transforms.Compose([
            transforms.Resize(config.image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=2
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=2
    )
    
    return train_loader, test_loader

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    config: CVConfig
) -> Dict[str, List[float]]:
    """Train the computer vision model."""
    
    model = model.to(config.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=3
    )
    
    history = {
        'train_loss': [],
        'test_loss': [],
        'train_acc': [],
        'test_acc': []
    }
    
    for epoch in range(config.num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for inputs, targets in train_loader:
            inputs = inputs.to(config.device)
            targets = targets.to(config.device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()
        
        # Testing phase
        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs = inputs.to(config.device)
                targets = targets.to(config.device)
                
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                test_total += targets.size(0)
                test_correct += predicted.eq(targets).sum().item()
        
        # Update learning rate
        scheduler.step(test_loss)
        
        # Record metrics
        history['train_loss'].append(train_loss / len(train_loader))
        history['test_loss'].append(test_loss / len(test_loader))
        history['train_acc'].append(train_correct / train_total)
        history['test_acc'].append(test_correct / test_total)
        
        print(f"Epoch {epoch+1}/{config.num_epochs}")
        print(f"Train Loss: {history['train_loss'][-1]:.4f}, "
              f"Train Acc: {history['train_acc'][-1]:.4f}")
        print(f"Test Loss: {history['test_loss'][-1]:.4f}, "
              f"Test Acc: {history['test_acc'][-1]:.4f}")
    
    return history

def demonstrate_transfer_learning():
    """Demonstrate transfer learning with ResNet."""
    
    # Load pre-trained ResNet
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    
    # Freeze all layers except the last one
    for param in model.parameters():
        param.requires_grad = False
    
    # Replace the last layer
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 10)
    
    return model

def demonstrate_object_detection():
    """Demonstrate object detection using pre-trained model."""
    
    # Load pre-trained model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    )
    model.eval()
    
    # Load and transform image
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    # Sample image (replace with actual image path)
    image = Image.open("sample_image.jpg")
    image_tensor = transform(image)
    
    # Perform detection
    with torch.no_grad():
        prediction = model([image_tensor])
    
    return prediction

def visualize_results(history: Dict[str, List[float]]):
    """Visualize training results."""
    
    plt.figure(figsize=(12, 4))
    
    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['test_loss'], label='Test Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['test_acc'], label='Test Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    return plt

if __name__ == '__main__':
    # Configuration
    config = CVConfig()
    
    # Prepare data
    train_loader, test_loader = prepare_data(config)
    
    # Create and train custom CNN
    model = ConvNet(config.num_classes)
    history = train_model(model, train_loader, test_loader, config)
    
    # Visualize results
    visualize_results(history)
    plt.show()
    
    # Demonstrate transfer learning
    transfer_model = demonstrate_transfer_learning()
    
    # Train transfer learning model
    transfer_history = train_model(
        transfer_model,
        train_loader,
        test_loader,
        config
    )
    
    # Visualize transfer learning results
    visualize_results(transfer_history)
    plt.show() 