"""
Demonstration of Bayesian methods and probabilistic programming.
"""

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass

@dataclass
class BayesianConfig:
    """Configuration for Bayesian analysis."""
    n_samples: int = 2000
    n_tune: int = 1000
    n_chains: int = 4
    random_seed: int = 42
    target_accept: float = 0.95

def generate_sample_data() -> Tuple[np.ndarray, np.ndarray]:
    """Generate sample data for Bayesian analysis."""
    
    np.random.seed(42)
    
    # True parameters
    true_alpha = 2.5
    true_beta = 1.8
    true_sigma = 0.5
    
    # Generate predictor variable
    n_samples = 100
    X = np.random.uniform(0, 10, n_samples)
    
    # Generate response variable with noise
    y = true_alpha + true_beta * X + np.random.normal(0, true_sigma, n_samples)
    
    return X, y

def linear_regression_pymc(
    X: np.ndarray,
    y: np.ndarray,
    config: BayesianConfig
) -> pm.Model:
    """Implement Bayesian linear regression using PyMC."""
    
    with pm.Model() as model:
        # Priors
        alpha = pm.Normal('alpha', mu=0, sigma=10)
        beta = pm.Normal('beta', mu=0, sigma=10)
        sigma = pm.HalfNormal('sigma', sigma=1)
        
        # Expected value
        mu = alpha + beta * X
        
        # Likelihood
        likelihood = pm.Normal('likelihood', mu=mu, sigma=sigma, observed=y)
        
        # Inference
        trace = pm.sample(
            draws=config.n_samples,
            tune=config.n_tune,
            chains=config.n_chains,
            random_seed=config.random_seed,
            target_accept=config.target_accept
        )
    
    return model, trace

def logistic_regression_pymc(
    X: np.ndarray,
    y: np.ndarray,
    config: BayesianConfig
) -> pm.Model:
    """Implement Bayesian logistic regression using PyMC."""
    
    with pm.Model() as model:
        # Priors
        alpha = pm.Normal('alpha', mu=0, sigma=10)
        beta = pm.Normal('beta', mu=0, sigma=10)
        
        # Expected probability
        mu = pm.math.sigmoid(alpha + beta * X)
        
        # Likelihood
        likelihood = pm.Bernoulli('likelihood', p=mu, observed=y)
        
        # Inference
        trace = pm.sample(
            draws=config.n_samples,
            tune=config.n_tune,
            chains=config.n_chains,
            random_seed=config.random_seed,
            target_accept=config.target_accept
        )
    
    return model, trace

def hierarchical_model_pymc(
    group_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
    config: BayesianConfig
) -> pm.Model:
    """Implement hierarchical Bayesian model using PyMC."""
    
    with pm.Model() as model:
        # Hyperpriors
        mu_alpha = pm.Normal('mu_alpha', mu=0, sigma=10)
        sigma_alpha = pm.HalfNormal('sigma_alpha', sigma=1)
        
        mu_beta = pm.Normal('mu_beta', mu=0, sigma=10)
        sigma_beta = pm.HalfNormal('sigma_beta', sigma=1)
        
        # Group-level parameters
        alpha = pm.Normal('alpha', mu=mu_alpha, sigma=sigma_alpha, shape=len(group_data))
        beta = pm.Normal('beta', mu=mu_beta, sigma=sigma_beta, shape=len(group_data))
        
        # Group-level standard deviations
        sigma = pm.HalfNormal('sigma', sigma=1, shape=len(group_data))
        
        # Likelihood for each group
        for i, (group, (X, y)) in enumerate(group_data.items()):
            mu = alpha[i] + beta[i] * X
            likelihood = pm.Normal(f'likelihood_{group}', mu=mu, sigma=sigma[i], observed=y)
        
        # Inference
        trace = pm.sample(
            draws=config.n_samples,
            tune=config.n_tune,
            chains=config.n_chains,
            random_seed=config.random_seed,
            target_accept=config.target_accept
        )
    
    return model, trace

def evaluate_model(trace: az.InferenceData) -> Dict[str, float]:
    """Evaluate Bayesian model using diagnostics."""
    
    summary = az.summary(trace)
    
    metrics = {
        'r_hat': summary['r_hat'].mean(),
        'ess_bulk': summary['ess_bulk'].mean(),
        'ess_tail': summary['ess_tail'].mean()
    }
    
    return metrics

def visualize_results(
    X: np.ndarray,
    y: np.ndarray,
    trace: az.InferenceData,
    model_type: str = 'linear'
):
    """Visualize Bayesian analysis results."""
    
    plt.figure(figsize=(15, 5))
    
    # Trace plots
    plt.subplot(1, 3, 1)
    az.plot_trace(trace, var_names=['alpha', 'beta'])
    plt.title('Parameter Traces')
    
    # Posterior distributions
    plt.subplot(1, 3, 2)
    az.plot_posterior(trace, var_names=['alpha', 'beta'])
    plt.title('Posterior Distributions')
    
    # Data and fit
    plt.subplot(1, 3, 3)
    plt.scatter(X, y, alpha=0.5)
    
    alpha_mean = trace.posterior['alpha'].mean().item()
    beta_mean = trace.posterior['beta'].mean().item()
    
    X_plot = np.linspace(X.min(), X.max(), 100)
    if model_type == 'linear':
        y_plot = alpha_mean + beta_mean * X_plot
    else:  # logistic
        y_plot = 1 / (1 + np.exp(-(alpha_mean + beta_mean * X_plot)))
    
    plt.plot(X_plot, y_plot, 'r-', label='Mean prediction')
    plt.title('Data and Model Fit')
    plt.legend()
    
    plt.tight_layout()
    return plt

if __name__ == '__main__':
    # Configuration
    config = BayesianConfig()
    
    # Generate data
    X, y = generate_sample_data()
    
    # Linear regression
    print("\nFitting Bayesian linear regression...")
    linear_model, linear_trace = linear_regression_pymc(X, y, config)
    
    # Generate binary data for logistic regression
    y_binary = (y > np.median(y)).astype(int)
    
    # Logistic regression
    print("\nFitting Bayesian logistic regression...")
    logistic_model, logistic_trace = logistic_regression_pymc(X, y_binary, config)
    
    # Generate grouped data for hierarchical model
    group_data = {
        'A': (X[:30], y[:30]),
        'B': (X[30:60], y[30:60]),
        'C': (X[60:], y[60:])
    }
    
    # Hierarchical model
    print("\nFitting hierarchical Bayesian model...")
    hierarchical_model, hierarchical_trace = hierarchical_model_pymc(group_data, config)
    
    # Evaluate models
    print("\nModel Diagnostics:")
    for model_name, trace in [
        ('Linear', linear_trace),
        ('Logistic', logistic_trace),
        ('Hierarchical', hierarchical_trace)
    ]:
        metrics = evaluate_model(trace)
        print(f"\n{model_name} Regression:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
    
    # Visualize results
    linear_plot = visualize_results(X, y, linear_trace, 'linear')
    linear_plot.suptitle('Bayesian Linear Regression')
    linear_plot.show()
    
    logistic_plot = visualize_results(X, y_binary, logistic_trace, 'logistic')
    logistic_plot.suptitle('Bayesian Logistic Regression')
    logistic_plot.show() 