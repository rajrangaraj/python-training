"""
Demonstration of advanced machine learning concepts, visualization, and model interpretation.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.inspection import permutation_importance
import shap
import eli5
from lime import lime_tabular
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass

@dataclass
class ModelInsights:
    """Class to store model interpretation results."""
    feature_importance: pd.Series
    permutation_importance: pd.DataFrame
    shap_values: np.ndarray
    lime_explanation: Any
    partial_dependence: Dict[str, np.ndarray]

def load_and_prepare_data() -> Tuple[pd.DataFrame, pd.Series]:
    """Load and prepare sample dataset with feature engineering."""
    from sklearn.datasets import load_breast_cancer
    
    # Load dataset
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name='target')
    
    # Feature engineering
    X = add_polynomial_features(X, degree=2, selected_features=['mean radius', 'mean texture'])
    X = add_interaction_features(X, ['mean radius', 'mean texture', 'mean perimeter'])
    
    return X, y

def add_polynomial_features(df: pd.DataFrame, degree: int, selected_features: List[str]) -> pd.DataFrame:
    """Add polynomial features for selected columns."""
    result = df.copy()
    
    for feature in selected_features:
        for d in range(2, degree + 1):
            result[f'{feature}_pow_{d}'] = df[feature] ** d
    
    return result

def add_interaction_features(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    """Add interaction features between selected columns."""
    result = df.copy()
    
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            name = f'{features[i]}_{features[j]}_interaction'
            result[name] = df[features[i]] * df[features[j]]
    
    return result

def plot_learning_curves(model, X: pd.DataFrame, y: pd.Series):
    """Plot learning curves to analyze model performance vs training size."""
    
    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y,
        train_sizes=np.linspace(0.1, 1.0, 10),
        cv=5,
        n_jobs=-1
    )
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, np.mean(train_scores, axis=1), label='Training score')
    plt.plot(train_sizes, np.mean(val_scores, axis=1), label='Cross-validation score')
    plt.xlabel('Training examples')
    plt.ylabel('Score')
    plt.title('Learning Curves')
    plt.legend(loc='best')
    plt.grid(True)
    
    return plt

def plot_roc_and_pr_curves(model, X_test: pd.DataFrame, y_test: pd.Series):
    """Plot ROC and Precision-Recall curves."""
    
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    pr_auc = auc(recall, precision)
    
    plt.figure(figsize=(15, 5))
    
    # Plot ROC curve
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    
    # Plot Precision-Recall curve
    plt.subplot(1, 2, 2)
    plt.plot(recall, precision, label=f'PR curve (AUC = {pr_auc:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    
    plt.tight_layout()
    return plt

def analyze_feature_importance(
    model,
    X: pd.DataFrame,
    y: pd.Series
) -> ModelInsights:
    """Analyze feature importance using multiple methods."""
    
    # Random Forest feature importance
    feature_importance = pd.Series(
        model.feature_importances_,
        index=X.columns
    ).sort_values(ascending=False)
    
    # Permutation importance
    perm_importance = permutation_importance(model, X, y, n_repeats=10)
    perm_importance_df = pd.DataFrame(
        {'importance_mean': perm_importance.importances_mean,
         'importance_std': perm_importance.importances_std},
        index=X.columns
    ).sort_values('importance_mean', ascending=False)
    
    # SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    
    # LIME explanation for a single instance
    explainer = lime_tabular.LimeTabularExplainer(
        X.values,
        feature_names=X.columns,
        class_names=['malignant', 'benign'],
        mode='classification'
    )
    lime_exp = explainer.explain_instance(
        X.iloc[0].values,
        model.predict_proba
    )
    
    # Partial dependence plots for top features
    top_features = feature_importance.head(3).index
    partial_dependence = {}
    for feature in top_features:
        values = np.linspace(X[feature].min(), X[feature].max(), 50)
        predictions = []
        for value in values:
            X_modified = X.copy()
            X_modified[feature] = value
            predictions.append(model.predict_proba(X_modified)[:, 1].mean())
        partial_dependence[feature] = (values, np.array(predictions))
    
    return ModelInsights(
        feature_importance=feature_importance,
        permutation_importance=perm_importance_df,
        shap_values=shap_values,
        lime_explanation=lime_exp,
        partial_dependence=partial_dependence
    )

def visualize_model_insights(insights: ModelInsights):
    """Visualize model interpretation results."""
    
    plt.figure(figsize=(20, 15))
    
    # Feature importance plot
    plt.subplot(3, 2, 1)
    insights.feature_importance.head(10).plot(kind='barh')
    plt.title('Random Forest Feature Importance')
    
    # Permutation importance plot
    plt.subplot(3, 2, 2)
    insights.permutation_importance.head(10)['importance_mean'].plot(kind='barh')
    plt.title('Permutation Feature Importance')
    
    # SHAP summary plot
    plt.subplot(3, 2, 3)
    shap.summary_plot(insights.shap_values, features=X, show=False)
    plt.title('SHAP Feature Importance')
    
    # LIME explanation plot
    plt.subplot(3, 2, 4)
    insights.lime_explanation.as_pyplot_figure()
    plt.title('LIME Local Explanation')
    
    # Partial dependence plots
    plt.subplot(3, 2, 5)
    for feature, (values, predictions) in insights.partial_dependence.items():
        plt.plot(values, predictions, label=feature)
    plt.title('Partial Dependence Plots')
    plt.legend()
    
    plt.tight_layout()
    return plt

if __name__ == '__main__':
    # Load and prepare data
    X, y = load_and_prepare_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create and train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Generate and plot learning curves
    learning_curves_plot = plot_learning_curves(model, X, y)
    learning_curves_plot.show()
    
    # Generate and plot ROC and PR curves
    performance_curves = plot_roc_and_pr_curves(model, X_test, y_test)
    performance_curves.show()
    
    # Analyze and visualize model insights
    insights = analyze_feature_importance(model, X, y)
    insights_plot = visualize_model_insights(insights)
    insights_plot.show() 