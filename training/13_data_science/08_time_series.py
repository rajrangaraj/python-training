"""
Demonstration of time series analysis and forecasting.
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta

@dataclass
class TimeSeriesConfig:
    """Configuration for time series analysis."""
    seasonality_period: int = 12  # Monthly data
    train_size: float = 0.8
    forecast_horizon: int = 12
    confidence_level: float = 0.95

def generate_sample_data() -> pd.DataFrame:
    """Generate sample time series data with trend, seasonality, and noise."""
    
    # Date range
    dates = pd.date_range(start='2018-01-01', end='2023-12-31', freq='D')
    
    # Components
    trend = np.linspace(0, 10, len(dates))
    seasonality = 5 * np.sin(2 * np.pi * np.arange(len(dates)) / 365.25)
    noise = np.random.normal(0, 1, len(dates))
    
    # Combine components
    values = trend + seasonality + noise
    
    # Create DataFrame
    df = pd.DataFrame({
        'ds': dates,
        'y': values
    })
    
    return df

def analyze_time_series(data: pd.DataFrame) -> Dict[str, Any]:
    """Perform time series analysis."""
    
    # Decomposition
    decomposition = seasonal_decompose(
        data['y'],
        period=365,
        extrapolate_trend='freq'
    )
    
    # Stationarity test
    adf_result = adfuller(data['y'])
    
    # Basic statistics
    stats = {
        'mean': data['y'].mean(),
        'std': data['y'].std(),
        'min': data['y'].min(),
        'max': data['y'].max()
    }
    
    # Rolling statistics
    rolling_mean = data['y'].rolling(window=30).mean()
    rolling_std = data['y'].rolling(window=30).std()
    
    return {
        'decomposition': decomposition,
        'adf_test': adf_result,
        'statistics': stats,
        'rolling_mean': rolling_mean,
        'rolling_std': rolling_std
    }

def train_arima_model(data: pd.DataFrame) -> Tuple[ARIMA, Dict[str, float]]:
    """Train ARIMA model for time series forecasting."""
    
    # Split data
    train_size = int(len(data) * 0.8)
    train = data[:train_size]
    test = data[train_size:]
    
    # Fit ARIMA model
    model = ARIMA(train['y'], order=(1, 1, 1))
    results = model.fit()
    
    # Make predictions
    predictions = results.predict(
        start=len(train),
        end=len(train) + len(test) - 1
    )
    
    # Calculate metrics
    metrics = {
        'mse': mean_squared_error(test['y'], predictions),
        'rmse': np.sqrt(mean_squared_error(test['y'], predictions)),
        'mae': mean_absolute_error(test['y'], predictions)
    }
    
    return results, metrics

def train_prophet_model(data: pd.DataFrame) -> Tuple[Prophet, Dict[str, float]]:
    """Train Facebook Prophet model for time series forecasting."""
    
    # Split data
    train_size = int(len(data) * 0.8)
    train = data[:train_size]
    test = data[train_size:]
    
    # Train Prophet model
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=True
    )
    model.fit(train)
    
    # Make predictions
    future = model.make_future_dataframe(periods=len(test))
    forecast = model.predict(future)
    
    # Calculate metrics
    predictions = forecast.tail(len(test))['yhat']
    metrics = {
        'mse': mean_squared_error(test['y'], predictions),
        'rmse': np.sqrt(mean_squared_error(test['y'], predictions)),
        'mae': mean_absolute_error(test['y'], predictions)
    }
    
    return model, metrics

def visualize_analysis(data: pd.DataFrame, analysis_results: Dict[str, Any]):
    """Visualize time series analysis results."""
    
    plt.figure(figsize=(15, 10))
    
    # Original time series
    plt.subplot(3, 1, 1)
    plt.plot(data['ds'], data['y'])
    plt.title('Original Time Series')
    
    # Decomposition
    decomposition = analysis_results['decomposition']
    plt.subplot(3, 1, 2)
    plt.plot(decomposition.trend, label='Trend')
    plt.plot(decomposition.seasonal, label='Seasonal')
    plt.title('Time Series Decomposition')
    plt.legend()
    
    # Rolling statistics
    plt.subplot(3, 1, 3)
    plt.plot(data['ds'], analysis_results['rolling_mean'], label='Rolling Mean')
    plt.plot(data['ds'], analysis_results['rolling_std'], label='Rolling Std')
    plt.title('Rolling Statistics')
    plt.legend()
    
    plt.tight_layout()
    return plt

def visualize_forecasts(
    data: pd.DataFrame,
    arima_forecast: pd.Series,
    prophet_forecast: pd.DataFrame
):
    """Visualize forecasting results."""
    
    plt.figure(figsize=(15, 10))
    
    # ARIMA forecast
    plt.subplot(2, 1, 1)
    plt.plot(data['ds'], data['y'], label='Actual')
    plt.plot(data['ds'].iloc[-len(arima_forecast):],
             arima_forecast, label='ARIMA Forecast')
    plt.title('ARIMA Forecast')
    plt.legend()
    
    # Prophet forecast
    plt.subplot(2, 1, 2)
    plt.plot(data['ds'], data['y'], label='Actual')
    plt.plot(prophet_forecast['ds'], prophet_forecast['yhat'],
             label='Prophet Forecast')
    plt.fill_between(
        prophet_forecast['ds'],
        prophet_forecast['yhat_lower'],
        prophet_forecast['yhat_upper'],
        alpha=0.3
    )
    plt.title('Prophet Forecast')
    plt.legend()
    
    plt.tight_layout()
    return plt

if __name__ == '__main__':
    # Generate sample data
    data = generate_sample_data()
    
    # Perform time series analysis
    analysis_results = analyze_time_series(data)
    
    # Train models
    arima_model, arima_metrics = train_arima_model(data)
    prophet_model, prophet_metrics = train_prophet_model(data)
    
    # Print metrics
    print("\nARIMA Metrics:")
    for metric, value in arima_metrics.items():
        print(f"{metric.upper()}: {value:.4f}")
    
    print("\nProphet Metrics:")
    for metric, value in prophet_metrics.items():
        print(f"{metric.upper()}: {value:.4f}")
    
    # Visualize results
    analysis_plot = visualize_analysis(data, analysis_results)
    analysis_plot.show()
    
    # Generate and visualize forecasts
    arima_forecast = arima_model.get_forecast(steps=30).predicted_mean
    future = prophet_model.make_future_dataframe(periods=30)
    prophet_forecast = prophet_model.predict(future)
    
    forecast_plot = visualize_forecasts(data, arima_forecast, prophet_forecast)
    forecast_plot.show() 