"""
Demonstration of PyTorch fundamentals and neural network implementation.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Any
from dataclasses import dataclass

@dataclass
class TrainingConfig:
    """Configuration for model training."""
    batch_size: int = 32
    learning_rate: float = 0.001
    num_epochs: int = 100
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

class CustomDataset(Dataset):
    """Custom PyTorch Dataset."""
    
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class NeuralNetwork(nn.Module):
    """Simple neural network architecture."""
    
    def __init__(self, input_size: int, hidden_sizes: List[int], output_size: int):
        super().__init__()
        
        layers = []
        prev_size = input_size
        
        # Create hidden layers
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_size),
                nn.Dropout(0.2)
            ])
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, output_size))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

def prepare_data() -> Tuple[np.ndarray, np.ndarray]:
    """Prepare sample dataset."""
    from sklearn.datasets import load_breast_cancer
    
    # Load dataset
    data = load_breast_cancer()
    X, y = data.data, data.target
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: TrainingConfig
) -> Dict[str, List[float]]:
    """Train the neural network."""
    
    model = model.to(config.device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
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
            loss = criterion(outputs.squeeze(), targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            predictions = (outputs.squeeze() > 0).float()
            train_correct += (predictions == targets).sum().item()
            train_total += targets.size(0)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(config.device)
                targets = targets.to(config.device)
                
                outputs = model(inputs)
                loss = criterion(outputs.squeeze(), targets)
                
                val_loss += loss.item()
                predictions = (outputs.squeeze() > 0).float()
                val_correct += (predictions == targets).sum().item()
                val_total += targets.size(0)
        
        # Record metrics
        history['train_loss'].append(train_loss / len(train_loader))
        history['val_loss'].append(val_loss / len(val_loader))
        history['train_acc'].append(train_correct / train_total)
        history['val_acc'].append(val_correct / val_total)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{config.num_epochs}")
            print(f"Train Loss: {history['train_loss'][-1]:.4f}, "
                  f"Train Acc: {history['train_acc'][-1]:.4f}")
            print(f"Val Loss: {history['val_loss'][-1]:.4f}, "
                  f"Val Acc: {history['val_acc'][-1]:.4f}")
    
    return history

def plot_training_history(history: Dict[str, List[float]]):
    """Plot training history."""
    
    plt.figure(figsize=(12, 4))
    
    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    return plt

if __name__ == '__main__':
    # Configuration
    config = TrainingConfig()
    
    # Prepare data
    X, y = prepare_data()
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Create datasets and dataloaders
    train_dataset = CustomDataset(X_train, y_train)
    val_dataset = CustomDataset(X_val, y_val)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size
    )
    
    # Create and train model
    model = NeuralNetwork(
        input_size=X.shape[1],
        hidden_sizes=[64, 32],
        output_size=1
    )
    
    history = train_model(model, train_loader, val_loader, config)
    
    # Plot results
    plot_training_history(history)
    plt.show() 