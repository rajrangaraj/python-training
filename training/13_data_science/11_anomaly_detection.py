"""
Demonstration of various anomaly detection techniques.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass

@dataclass
class AnomalyConfig:
    """Configuration for anomaly detection."""
    contamination: float = 0.1
    random_state: int = 42
    n_neighbors: int = 20
    kernel: str = 'rbf'
    nu: float = 0.1

def generate_sample_data() -> Tuple[np.ndarray, np.ndarray]:
    """Generate sample data with anomalies."""
    
    # Generate normal data
    n_samples = 1000
    n_outliers = int(n_samples * 0.1)
    n_features = 2
    
    # Generate normal points
    X = np.random.randn(n_samples - n_outliers, n_features)
    
    # Generate outliers
    outliers = np.random.uniform(
        low=-4,
        high=4,
        size=(n_outliers, n_features)
    )
    
    # Combine normal and outlier points
    X = np.vstack([X, outliers])
    
    # True labels (0: normal, 1: anomaly)
    y = np.zeros(n_samples)
    y[-n_outliers:] = 1
    
    return X, y

def detect_anomalies(
    X: np.ndarray,
    config: AnomalyConfig
) -> Dict[str, np.ndarray]:
    """Apply different anomaly detection methods."""
    
    # Standardize data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Isolation Forest
    iso_forest = IsolationForest(
        contamination=config.contamination,
        random_state=config.random_state
    )
    iso_forest_labels = iso_forest.fit_predict(X_scaled)
    
    # One-Class SVM
    one_class_svm = OneClassSVM(
        kernel=config.kernel,
        nu=config.nu
    )
    one_class_svm_labels = one_class_svm.fit_predict(X_scaled)
    
    # Local Outlier Factor
    lof = LocalOutlierFactor(
        n_neighbors=config.n_neighbors,
        contamination=config.contamination
    )
    lof_labels = lof.fit_predict(X_scaled)
    
    # Robust Covariance
    robust_cov = EllipticEnvelope(
        contamination=config.contamination,
        random_state=config.random_state
    )
    robust_cov_labels = robust_cov.fit_predict(X_scaled)
    
    # Convert predictions to binary (0: normal, 1: anomaly)
    results = {
        'isolation_forest': (iso_forest_labels == -1).astype(int),
        'one_class_svm': (one_class_svm_labels == -1).astype(int),
        'local_outlier_factor': (lof_labels == -1).astype(int),
        'robust_covariance': (robust_cov_labels == -1).astype(int)
    }
    
    return results

def evaluate_detection(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:
    """Evaluate anomaly detection results."""
    
    from sklearn.metrics import (
        precision_score,
        recall_score,
        f1_score,
        accuracy_score
    )
    
    metrics = {
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
        'accuracy': accuracy_score(y_true, y_pred)
    }
    
    return metrics

def visualize_results(
    X: np.ndarray,
    y_true: np.ndarray,
    detection_results: Dict[str, np.ndarray]
):
    """Visualize anomaly detection results."""
    
    plt.figure(figsize=(20, 5))
    
    # Plot original data
    plt.subplot(1, 5, 1)
    plt.scatter(X[:, 0], X[:, 1], c=y_true, cmap='viridis')
    plt.title('Original Data')
    
    # Plot detection results
    for idx, (method, labels) in enumerate(detection_results.items(), 2):
        plt.subplot(1, 5, idx)
        plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
        plt.title(f'{method.replace("_", " ").title()}')
    
    plt.tight_layout()
    return plt

def analyze_feature_importance(
    X: np.ndarray,
    detection_results: Dict[str, np.ndarray]
) -> Dict[str, np.ndarray]:
    """Analyze feature importance for anomaly detection."""
    
    feature_importance = {}
    
    # For each detection method
    for method, labels in detection_results.items():
        # Calculate mean difference between normal and anomaly points
        normal_points = X[labels == 0]
        anomaly_points = X[labels == 1]
        
        normal_mean = np.mean(normal_points, axis=0)
        anomaly_mean = np.mean(anomaly_points, axis=0)
        
        # Feature importance as absolute difference in means
        importance = np.abs(normal_mean - anomaly_mean)
        feature_importance[method] = importance
    
    return feature_importance

def visualize_feature_importance(
    feature_importance: Dict[str, np.ndarray]
):
    """Visualize feature importance for each method."""
    
    plt.figure(figsize=(12, 6))
    
    methods = list(feature_importance.keys())
    n_features = len(feature_importance[methods[0]])
    
    x = np.arange(n_features)
    width = 0.8 / len(methods)
    
    for i, (method, importance) in enumerate(feature_importance.items()):
        plt.bar(
            x + i * width,
            importance,
            width,
            label=method.replace('_', ' ').title()
        )
    
    plt.xlabel('Feature Index')
    plt.ylabel('Importance Score')
    plt.title('Feature Importance by Detection Method')
    plt.legend()
    plt.tight_layout()
    
    return plt

if __name__ == '__main__':
    # Configuration
    config = AnomalyConfig()
    
    # Generate data
    X, true_labels = generate_sample_data()
    
    # Detect anomalies
    detection_results = detect_anomalies(X, config)
    
    # Evaluate results
    print("\nAnomaly Detection Evaluation:")
    for method, labels in detection_results.items():
        metrics = evaluate_detection(true_labels, labels)
        print(f"\n{method.replace('_', ' ').title()}:")
        for metric_name, score in metrics.items():
            print(f"{metric_name}: {score:.4f}")
    
    # Analyze feature importance
    importance = analyze_feature_importance(X, detection_results)
    
    # Visualize results
    detection_plot = visualize_results(X, true_labels, detection_results)
    detection_plot.show()
    
    importance_plot = visualize_feature_importance(importance)
    importance_plot.show() 