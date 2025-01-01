"""
Demonstration of clustering algorithms and dimensionality reduction techniques.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA, NMF
from sklearn.manifold import TSNE, MDS
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass

@dataclass
class ClusteringConfig:
    """Configuration for clustering analysis."""
    n_clusters: int = 5
    n_components: int = 2
    random_state: int = 42
    perplexity: float = 30.0
    eps: float = 0.5
    min_samples: int = 5

def generate_sample_data() -> Tuple[np.ndarray, np.ndarray]:
    """Generate sample data with clusters."""
    
    # Generate clusters
    n_samples = 1000
    centers = [
        (0, 0), (5, 5), (-3, 3), (3, -3), (-5, -5)
    ]
    
    X = np.vstack([
        np.random.randn(n_samples // 5, 2) + center
        for center in centers
    ])
    
    # Add noise features
    noise = np.random.randn(n_samples, 3)
    X = np.hstack([X, noise])
    
    # Generate labels for evaluation
    y = np.repeat(range(5), n_samples // 5)
    
    return X, y

def apply_dimensionality_reduction(
    X: np.ndarray,
    config: ClusteringConfig
) -> Dict[str, np.ndarray]:
    """Apply different dimensionality reduction techniques."""
    
    # Standardize data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # PCA
    pca = PCA(n_components=config.n_components)
    X_pca = pca.fit_transform(X_scaled)
    
    # t-SNE
    tsne = TSNE(
        n_components=config.n_components,
        perplexity=config.perplexity,
        random_state=config.random_state
    )
    X_tsne = tsne.fit_transform(X_scaled)
    
    # MDS
    mds = MDS(
        n_components=config.n_components,
        random_state=config.random_state
    )
    X_mds = mds.fit_transform(X_scaled)
    
    # NMF (for non-negative data)
    X_pos = X_scaled - X_scaled.min()
    nmf = NMF(
        n_components=config.n_components,
        random_state=config.random_state
    )
    X_nmf = nmf.fit_transform(X_pos)
    
    return {
        'pca': X_pca,
        'tsne': X_tsne,
        'mds': X_mds,
        'nmf': X_nmf,
        'explained_variance': pca.explained_variance_ratio_
    }

def apply_clustering(
    X: np.ndarray,
    config: ClusteringConfig
) -> Dict[str, Any]:
    """Apply different clustering algorithms."""
    
    # K-means
    kmeans = KMeans(
        n_clusters=config.n_clusters,
        random_state=config.random_state
    )
    kmeans_labels = kmeans.fit_predict(X)
    
    # DBSCAN
    dbscan = DBSCAN(
        eps=config.eps,
        min_samples=config.min_samples
    )
    dbscan_labels = dbscan.fit_predict(X)
    
    # Hierarchical Clustering
    hierarchical = AgglomerativeClustering(
        n_clusters=config.n_clusters
    )
    hierarchical_labels = hierarchical.fit_predict(X)
    
    # Gaussian Mixture
    gmm = GaussianMixture(
        n_components=config.n_clusters,
        random_state=config.random_state
    )
    gmm_labels = gmm.fit_predict(X)
    
    return {
        'kmeans': kmeans_labels,
        'dbscan': dbscan_labels,
        'hierarchical': hierarchical_labels,
        'gmm': gmm_labels,
        'kmeans_centers': kmeans.cluster_centers_,
        'gmm_means': gmm.means_
    }

def evaluate_clustering(
    X: np.ndarray,
    labels: np.ndarray
) -> Dict[str, float]:
    """Evaluate clustering results."""
    
    metrics = {
        'silhouette': silhouette_score(X, labels),
        'calinski_harabasz': calinski_harabasz_score(X, labels)
    }
    
    return metrics

def visualize_results(
    X_reduced: Dict[str, np.ndarray],
    clustering_results: Dict[str, np.ndarray],
    true_labels: np.ndarray
):
    """Visualize dimensionality reduction and clustering results."""
    
    plt.figure(figsize=(20, 15))
    
    # Plot dimensionality reduction results
    reduction_methods = ['pca', 'tsne', 'mds', 'nmf']
    for idx, method in enumerate(reduction_methods, 1):
        plt.subplot(3, 3, idx)
        plt.scatter(
            X_reduced[method][:, 0],
            X_reduced[method][:, 1],
            c=true_labels,
            cmap='viridis'
        )
        plt.title(f'{method.upper()} Projection')
    
    # Plot clustering results
    clustering_methods = ['kmeans', 'dbscan', 'hierarchical', 'gmm']
    for idx, method in enumerate(clustering_methods, 5):
        plt.subplot(3, 3, idx)
        plt.scatter(
            X_reduced['pca'][:, 0],
            X_reduced['pca'][:, 1],
            c=clustering_results[method],
            cmap='viridis'
        )
        plt.title(f'{method.upper()} Clustering')
    
    # Plot explained variance ratio for PCA
    plt.subplot(3, 3, 9)
    plt.plot(
        np.cumsum(X_reduced['explained_variance']),
        'bo-'
    )
    plt.title('PCA Explained Variance Ratio')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    
    plt.tight_layout()
    return plt

if __name__ == '__main__':
    # Configuration
    config = ClusteringConfig()
    
    # Generate data
    X, true_labels = generate_sample_data()
    
    # Apply dimensionality reduction
    reduced_data = apply_dimensionality_reduction(X, config)
    
    # Apply clustering
    clustering_results = apply_clustering(X, config)
    
    # Evaluate clustering
    print("\nClustering Evaluation:")
    for method, labels in clustering_results.items():
        if method not in ['kmeans_centers', 'gmm_means']:
            metrics = evaluate_clustering(X, labels)
            print(f"\n{method.upper()}:")
            for metric_name, score in metrics.items():
                print(f"{metric_name}: {score:.4f}")
    
    # Visualize results
    visualization = visualize_results(
        reduced_data,
        clustering_results,
        true_labels
    )
    visualization.show() 