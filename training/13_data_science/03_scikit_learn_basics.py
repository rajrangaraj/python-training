"""
Demonstration of Scikit-learn fundamentals and machine learning pipelines.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass

@dataclass
class ModelResult:
    """Class to store model evaluation results."""
    name: str
    accuracy: float
    cv_scores: np.ndarray
    confusion_matrix: np.ndarray
    classification_report: str
    best_params: Dict[str, Any] = None

def load_sample_data() -> Tuple[pd.DataFrame, pd.Series]:
    """Load and prepare sample dataset."""
    from sklearn.datasets import load_breast_cancer
    
    # Load dataset
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name='target')
    
    # Add some missing values for demonstration
    mask = np.random.random(X.shape) < 0.1
    X[mask] = np.nan
    
    return X, y

def create_preprocessing_pipeline(numeric_features: List[str]) -> ColumnTransformer:
    """Create a preprocessing pipeline for numeric features."""
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features)
        ]
    )
    
    return preprocessor

def train_and_evaluate_model(
    model,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    model_name: str,
    param_grid: Dict[str, List[Any]] = None
) -> ModelResult:
    """Train, optimize, and evaluate a model."""
    
    if param_grid:
        # Perform grid search for hyperparameter optimization
        grid_search = GridSearchCV(
            model,
            param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        model = grid_search.best_estimator_
        best_params = grid_search.best_params_
    else:
        # Train model with default parameters
        model.fit(X_train, y_train)
        best_params = None
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    
    return ModelResult(
        name=model_name,
        accuracy=accuracy,
        cv_scores=cv_scores,
        confusion_matrix=conf_matrix,
        classification_report=class_report,
        best_params=best_params
    )

def compare_models(X: pd.DataFrame, y: pd.Series) -> Dict[str, ModelResult]:
    """Compare different models on the same dataset."""
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Create preprocessing pipeline
    preprocessor = create_preprocessing_pipeline(X.columns.tolist())
    
    # Define models with their parameter grids
    models = {
        'logistic': (
            LogisticRegression(random_state=42),
            {'C': [0.1, 1.0, 10.0], 'max_iter': [1000]}
        ),
        'decision_tree': (
            DecisionTreeClassifier(random_state=42),
            {'max_depth': [3, 5, 7], 'min_samples_split': [2, 5, 10]}
        ),
        'random_forest': (
            RandomForestClassifier(random_state=42),
            {'n_estimators': [100, 200], 'max_depth': [5, 10]}
        ),
        'svm': (
            SVC(random_state=42),
            {'C': [0.1, 1.0, 10.0], 'kernel': ['rbf', 'linear']}
        )
    }
    
    results = {}
    
    # Train and evaluate each model
    for name, (model, param_grid) in models.items():
        # Create pipeline with preprocessor and model
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
        
        # Update parameter grid with pipeline prefix
        pipeline_param_grid = {
            f'classifier__{key}': value 
            for key, value in param_grid.items()
        }
        
        # Train and evaluate
        result = train_and_evaluate_model(
            pipeline,
            X_train,
            X_test,
            y_train,
            y_test,
            name,
            pipeline_param_grid
        )
        
        results[name] = result
    
    return results

def visualize_results(results: Dict[str, ModelResult]):
    """Visualize model comparison results."""
    
    plt.figure(figsize=(15, 10))
    
    # Accuracy comparison
    plt.subplot(2, 2, 1)
    accuracies = [result.accuracy for result in results.values()]
    plt.bar(results.keys(), accuracies)
    plt.title('Model Accuracy Comparison')
    plt.xticks(rotation=45)
    
    # Cross-validation scores
    plt.subplot(2, 2, 2)
    plt.boxplot([result.cv_scores for result in results.values()],
                labels=results.keys())
    plt.title('Cross-validation Scores')
    plt.xticks(rotation=45)
    
    # Confusion matrices
    plt.subplot(2, 2, 3)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for (name, result), ax in zip(results.items(), axes.ravel()):
        sns.heatmap(result.confusion_matrix, annot=True, fmt='d', ax=ax)
        ax.set_title(f'{name} Confusion Matrix')
    
    plt.tight_layout()
    return plt

if __name__ == '__main__':
    # Load and prepare data
    X, y = load_sample_data()
    print("\nDataset shape:", X.shape)
    print("\nFeature names:", X.columns.tolist())
    
    # Compare models
    results = compare_models(X, y)
    
    # Print results
    print("\nModel Comparison Results:")
    for name, result in results.items():
        print(f"\n{'-'*50}")
        print(f"\nModel: {name}")
        print(f"Accuracy: {result.accuracy:.4f}")
        print(f"CV Scores: mean={result.cv_scores.mean():.4f} (+/- {result.cv_scores.std()*2:.4f})")
        print(f"Best Parameters: {result.best_params}")
        print("\nClassification Report:")
        print(result.classification_report)
    
    # Visualize results
    visualize_results(results)
    plt.show() 