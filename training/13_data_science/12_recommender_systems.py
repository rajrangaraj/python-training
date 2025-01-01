"""
Demonstration of recommender system implementations.
"""

import numpy as np
import pandas as pd
from scipy.sparse.linalg import svds
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns

@dataclass
class RecommenderConfig:
    """Configuration for recommender systems."""
    n_factors: int = 50
    n_epochs: int = 20
    learning_rate: float = 0.001
    batch_size: int = 64
    reg_factor: float = 0.01
    random_state: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

class MatrixFactorization(nn.Module):
    """Neural network based matrix factorization."""
    
    def __init__(self, n_users: int, n_items: int, n_factors: int):
        super().__init__()
        
        # User and item embeddings
        self.user_factors = nn.Embedding(n_users, n_factors)
        self.item_factors = nn.Embedding(n_items, n_factors)
        
        # Initialize weights
        self.user_factors.weight.data.normal_(0, 0.1)
        self.item_factors.weight.data.normal_(0, 0.1)
    
    def forward(self, user: torch.Tensor, item: torch.Tensor) -> torch.Tensor:
        # Look up embeddings
        user_embedding = self.user_factors(user)
        item_embedding = self.item_factors(item)
        
        # Compute dot product
        return (user_embedding * item_embedding).sum(dim=1)

def generate_sample_data() -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    """Generate sample user-item interaction data."""
    
    np.random.seed(42)
    
    # Generate users and items
    n_users = 1000
    n_items = 500
    n_interactions = 10000
    
    # Generate random interactions
    users = np.random.randint(0, n_users, n_interactions)
    items = np.random.randint(0, n_items, n_interactions)
    ratings = np.random.normal(3.5, 1.0, n_interactions)
    ratings = np.clip(ratings, 1, 5)
    
    # Create DataFrame
    df = pd.DataFrame({
        'user_id': users,
        'item_id': items,
        'rating': ratings
    })
    
    # Generate item metadata
    categories = ['A', 'B', 'C', 'D']
    item_metadata = {
        'category': [np.random.choice(categories) for _ in range(n_items)],
        'tags': [
            [f'tag_{i}' for i in np.random.choice(10, 3)]
            for _ in range(n_items)
        ]
    }
    
    return df, item_metadata

def collaborative_filtering(
    df: pd.DataFrame,
    config: RecommenderConfig
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Implement collaborative filtering using SVD."""
    
    # Create user-item matrix
    user_item_matrix = df.pivot(
        index='user_id',
        columns='item_id',
        values='rating'
    ).fillna(0)
    
    # Normalize the data
    user_ratings_mean = np.mean(user_item_matrix, axis=1)
    matrix_norm = user_item_matrix - user_ratings_mean.reshape(-1, 1)
    
    # Apply SVD
    U, sigma, Vt = svds(matrix_norm, k=config.n_factors)
    
    # Convert to diagonal matrix
    sigma = np.diag(sigma)
    
    return U, sigma, Vt

def content_based_filtering(
    df: pd.DataFrame,
    item_metadata: Dict[str, List[str]]
) -> np.ndarray:
    """Implement content-based filtering."""
    
    # Create item features
    n_items = len(item_metadata['category'])
    
    # One-hot encode categories
    categories = pd.get_dummies(item_metadata['category'])
    
    # Create tag features (multi-hot encoding)
    all_tags = set([
        tag for tags in item_metadata['tags']
        for tag in tags
    ])
    tag_matrix = np.zeros((n_items, len(all_tags)))
    
    for i, tags in enumerate(item_metadata['tags']):
        for tag in tags:
            tag_matrix[i, list(all_tags).index(tag)] = 1
    
    # Combine features
    item_features = np.hstack([categories, tag_matrix])
    
    # Calculate item similarities
    item_similarities = cosine_similarity(item_features)
    
    return item_similarities

def train_neural_collaborative_filtering(
    df: pd.DataFrame,
    config: RecommenderConfig
) -> MatrixFactorization:
    """Train neural network based collaborative filtering."""
    
    # Create tensors
    users = torch.LongTensor(df['user_id'].values)
    items = torch.LongTensor(df['item_id'].values)
    ratings = torch.FloatTensor(df['rating'].values)
    
    # Create model
    n_users = df['user_id'].nunique()
    n_items = df['item_id'].nunique()
    model = MatrixFactorization(n_users, n_items, config.n_factors)
    model = model.to(config.device)
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.reg_factor
    )
    
    # Training loop
    model.train()
    n_batches = len(df) // config.batch_size
    
    for epoch in range(config.n_epochs):
        total_loss = 0
        
        for i in range(n_batches):
            # Get batch
            start_idx = i * config.batch_size
            end_idx = start_idx + config.batch_size
            
            user_batch = users[start_idx:end_idx].to(config.device)
            item_batch = items[start_idx:end_idx].to(config.device)
            rating_batch = ratings[start_idx:end_idx].to(config.device)
            
            # Forward pass
            predictions = model(user_batch, item_batch)
            loss = criterion(predictions, rating_batch)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Print progress
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch + 1}/{config.n_epochs}")
            print(f"Average Loss: {total_loss / n_batches:.4f}")
    
    return model

def evaluate_recommendations(
    predictions: np.ndarray,
    actual: np.ndarray
) -> Dict[str, float]:
    """Evaluate recommendation quality."""
    
    from sklearn.metrics import (
        mean_squared_error,
        mean_absolute_error,
        r2_score
    )
    
    metrics = {
        'rmse': np.sqrt(mean_squared_error(actual, predictions)),
        'mae': mean_absolute_error(actual, predictions),
        'r2': r2_score(actual, predictions)
    }
    
    return metrics

def visualize_results(
    df: pd.DataFrame,
    collaborative_predictions: np.ndarray,
    content_similarities: np.ndarray
):
    """Visualize recommendation results."""
    
    plt.figure(figsize=(15, 5))
    
    # Rating distribution
    plt.subplot(1, 3, 1)
    sns.histplot(data=df, x='rating', bins=20)
    plt.title('Rating Distribution')
    
    # Collaborative filtering heatmap
    plt.subplot(1, 3, 2)
    sns.heatmap(
        collaborative_predictions[:10, :10],
        cmap='viridis'
    )
    plt.title('Collaborative Filtering\nPredictions Sample')
    
    # Content-based similarity heatmap
    plt.subplot(1, 3, 3)
    sns.heatmap(
        content_similarities[:10, :10],
        cmap='viridis'
    )
    plt.title('Content-based\nSimilarities Sample')
    
    plt.tight_layout()
    return plt

if __name__ == '__main__':
    # Configuration
    config = RecommenderConfig()
    
    # Generate data
    df, item_metadata = generate_sample_data()
    
    # Collaborative filtering
    U, sigma, Vt = collaborative_filtering(df, config)
    cf_predictions = np.dot(np.dot(U, sigma), Vt)
    
    # Content-based filtering
    content_similarities = content_based_filtering(df, item_metadata)
    
    # Neural collaborative filtering
    ncf_model = train_neural_collaborative_filtering(df, config)
    
    # Evaluate results
    print("\nCollaborative Filtering Evaluation:")
    cf_metrics = evaluate_recommendations(
        cf_predictions.flatten(),
        df['rating'].values
    )
    for metric, value in cf_metrics.items():
        print(f"{metric.upper()}: {value:.4f}")
    
    # Visualize results
    visualization = visualize_results(
        df,
        cf_predictions,
        content_similarities
    )
    visualization.show() 