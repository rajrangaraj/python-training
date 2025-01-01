"""
Demonstration of Graph Neural Networks using PyTorch Geometric.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from torch_geometric.data import Data, DataLoader
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from sklearn.manifold import TSNE

@dataclass
class GNNConfig:
    """Configuration for Graph Neural Networks."""
    hidden_channels: int = 64
    num_layers: int = 2
    dropout_rate: float = 0.5
    learning_rate: float = 0.01
    num_epochs: int = 200
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    random_seed: int = 42

class GCN(nn.Module):
    """Graph Convolutional Network."""
    
    def __init__(self, num_features: int, hidden_channels: int, num_classes: int, dropout_rate: float):
        super().__init__()
        
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, num_classes)
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return x

class GAT(nn.Module):
    """Graph Attention Network."""
    
    def __init__(self, num_features: int, hidden_channels: int, num_classes: int, dropout_rate: float):
        super().__init__()
        
        self.conv1 = GATConv(num_features, hidden_channels, heads=8, dropout=dropout_rate)
        self.conv2 = GATConv(hidden_channels * 8, num_classes, heads=1, concat=False, dropout=dropout_rate)
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return x

class GraphSAGE(nn.Module):
    """GraphSAGE model."""
    
    def __init__(self, num_features: int, hidden_channels: int, num_classes: int, dropout_rate: float):
        super().__init__()
        
        self.conv1 = SAGEConv(num_features, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, num_classes)
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return x

def train_model(
    model: nn.Module,
    data: Data,
    config: GNNConfig
) -> Dict[str, List[float]]:
    """Train GNN model."""
    
    model = model.to(config.device)
    data = data.to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
    }
    
    for epoch in range(config.num_epochs):
        # Training
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        
        # Evaluation
        model.eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index)
            
            # Training metrics
            train_loss = criterion(out[data.train_mask], data.y[data.train_mask]).item()
            train_acc = (out[data.train_mask].argmax(dim=1) == data.y[data.train_mask]).float().mean().item()
            
            # Validation metrics
            val_loss = criterion(out[data.val_mask], data.y[data.val_mask]).item()
            val_acc = (out[data.val_mask].argmax(dim=1) == data.y[data.val_mask]).float().mean().item()
        
        # Record metrics
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{config.num_epochs}")
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    
    return history

def evaluate_model(
    model: nn.Module,
    data: Data
) -> Dict[str, float]:
    """Evaluate GNN model."""
    
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)
        
        test_acc = (pred[data.test_mask] == data.y[data.test_mask]).float().mean().item()
    
    return {'test_accuracy': test_acc}

def visualize_graph(
    data: Data,
    embeddings: torch.Tensor,
    title: str = "Graph Visualization"
):
    """Visualize graph structure and node embeddings."""
    
    plt.figure(figsize=(15, 5))
    
    # Graph structure
    plt.subplot(1, 3, 1)
    G = nx.Graph()
    edge_index = data.edge_index.cpu().numpy()
    edges = list(zip(edge_index[0], edge_index[1]))
    G.add_edges_from(edges)
    
    pos = nx.spring_layout(G)
    nx.draw(G, pos, node_size=50, node_color='lightblue',
            with_labels=False, alpha=0.8)
    plt.title("Graph Structure")
    
    # Node embeddings
    plt.subplot(1, 3, 2)
    tsne = TSNE(n_components=2, random_state=42)
    node_embeddings = embeddings.detach().cpu().numpy()
    node_embeddings_2d = tsne.fit_transform(node_embeddings)
    
    plt.scatter(
        node_embeddings_2d[:, 0],
        node_embeddings_2d[:, 1],
        c=data.y.cpu().numpy(),
        cmap='Set3',
        alpha=0.8
    )
    plt.title("Node Embeddings (t-SNE)")
    
    # Training history
    plt.subplot(1, 3, 3)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.suptitle(title)
    plt.tight_layout()
    return plt

if __name__ == '__main__':
    # Configuration
    config = GNNConfig()
    
    # Load dataset (Cora citation network)
    dataset = Planetoid(root='data/Cora', name='Cora', transform=NormalizeFeatures())
    data = dataset[0]
    
    # Create models
    models = {
        'GCN': GCN(dataset.num_features, config.hidden_channels, dataset.num_classes, config.dropout_rate),
        'GAT': GAT(dataset.num_features, config.hidden_channels, dataset.num_classes, config.dropout_rate),
        'GraphSAGE': GraphSAGE(dataset.num_features, config.hidden_channels, dataset.num_classes, config.dropout_rate)
    }
    
    # Train and evaluate models
    results = {}
    for name, model in models.items():
        print(f"\nTraining {name}...")
        history = train_model(model, data, config)
        metrics = evaluate_model(model, data)
        
        print(f"\n{name} Test Accuracy: {metrics['test_accuracy']:.4f}")
        
        # Get node embeddings
        model.eval()
        with torch.no_grad():
            embeddings = model(data.x, data.edge_index)
        
        # Visualize results
        viz = visualize_graph(data, embeddings, f"{name} Results")
        viz.show()
        
        results[name] = {
            'history': history,
            'metrics': metrics,
            'embeddings': embeddings
        } 