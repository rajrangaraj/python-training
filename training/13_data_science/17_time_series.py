"""
Demonstration of time series analysis and forecasting techniques.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet

@dataclass
class TimeSeriesConfig:
    """Configuration for time series analysis."""
    sequence_length: int = 10
    forecast_horizon: int = 5
    hidden_size: int = 64
    num_layers: int = 2
    batch_size: int = 32
    learning_rate: float = 0.001
    num_epochs: int = 100
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    random_seed: int = 42

class TimeSeriesDataset(Dataset):
    """Custom dataset for time series data."""
    
    def __init__(self, data: np.ndarray, sequence_length: int, forecast_horizon: int):
        self.data = torch.FloatTensor(data)
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon
    
    def __len__(self) -> int:
        return len(self.data) - self.sequence_length - self.forecast_horizon + 1
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.data[idx:idx + self.sequence_length]
        y = self.data[idx + self.sequence_length:idx + self.sequence_length + self.forecast_horizon]
        return x, y

class LSTM(nn.Module):
    """LSTM model for time series forecasting."""
    
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, output_size: int):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2
        )
        
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

def generate_sample_data() -> pd.DataFrame:
    """Generate sample time series data."""
    
    np.random.seed(42)
    
    # Time index
    dates = pd.date_range(start='2020-01-01', end='2022-12-31', freq='D')
    n = len(dates)
    
    # Trend component
    trend = np.linspace(0, 2, n)
    
    # Seasonal component (yearly)
    seasonal = 0.5 * np.sin(2 * np.pi * np.arange(n) / 365.25)
    
    # Random noise
    noise = np.random.normal(0, 0.1, n)
    
    # Combine components
    values = 10 + trend + seasonal + noise
    
    return pd.DataFrame({
        'date': dates,
        'value': values
    })

def analyze_time_series(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze time series characteristics."""
    
    analysis = {}
    
    # Basic statistics
    analysis['statistics'] = df['value'].describe()
    
    # Decomposition
    decomposition = seasonal_decompose(
        df['value'],
        period=365,
        extrapolate_trend='freq'
    )
    
    analysis['decomposition'] = {
        'trend': decomposition.trend,
        'seasonal': decomposition.seasonal,
        'residual': decomposition.resid
    }
    
    # Stationarity test (Augmented Dickey-Fuller)
    adf_result = adfuller(df['value'].dropna())
    analysis['stationarity'] = {
        'test_statistic': adf_result[0],
        'p_value': adf_result[1],
        'critical_values': adf_result[4]
    }
    
    return analysis

def train_lstm(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: TimeSeriesConfig
) -> Dict[str, List[float]]:
    """Train LSTM model."""
    
    model = model.to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.MSELoss()
    
    history = {
        'train_loss': [],
        'val_loss': []
    }
    
    for epoch in range(config.num_epochs):
        # Training
        model.train()
        train_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(config.device)
            batch_y = batch_y.to(config.device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(config.device)
                batch_y = batch_y.to(config.device)
                
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
        
        # Record metrics
        history['train_loss'].append(train_loss / len(train_loader))
        history['val_loss'].append(val_loss / len(val_loader))
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{config.num_epochs}]")
            print(f"Train Loss: {history['train_loss'][-1]:.4f}")
            print(f"Val Loss: {history['val_loss'][-1]:.4f}")
    
    return history

def visualize_results(
    df: pd.DataFrame,
    analysis: Dict[str, Any],
    history: Dict[str, List[float]],
    predictions: np.ndarray
):
    """Visualize time series analysis results."""
    
    plt.figure(figsize=(15, 10))
    
    # Original time series
    plt.subplot(2, 2, 1)
    plt.plot(df['date'], df['value'])
    plt.title('Original Time Series')
    plt.xticks(rotation=45)
    
    # Decomposition
    plt.subplot(2, 2, 2)
    plt.plot(analysis['decomposition']['trend'], label='Trend')
    plt.plot(analysis['decomposition']['seasonal'], label='Seasonal')
    plt.title('Time Series Decomposition')
    plt.legend()
    
    # Training history
    plt.subplot(2, 2, 3)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Predictions
    plt.subplot(2, 2, 4)
    plt.plot(df['value'].values[-100:], label='Actual')
    plt.plot(range(len(df)-50, len(df)+50), predictions, label='Predicted')
    plt.title('Model Predictions')
    plt.legend()
    
    plt.tight_layout()
    return plt

if __name__ == '__main__':
    # Configuration
    config = TimeSeriesConfig()
    
    # Generate data
    df = generate_sample_data()
    
    # Analyze time series
    analysis = analyze_time_series(df)
    
    # Data preprocessing
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df['value'].values.reshape(-1, 1))
    
    # Create datasets
    dataset = TimeSeriesDataset(scaled_data, config.sequence_length, config.forecast_horizon)
    train_size = int(0.8 * len(dataset))
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, len(dataset) - train_size])
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size)
    
    # Create and train LSTM model
    model = LSTM(
        input_size=1,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        output_size=config.forecast_horizon
    )
    
    history = train_lstm(model, train_loader, val_loader, config)
    
    # Generate predictions
    model.eval()
    with torch.no_grad():
        x = torch.FloatTensor(scaled_data[-config.sequence_length:]).unsqueeze(0)
        predictions = model(x).numpy()
        predictions = scaler.inverse_transform(predictions.reshape(-1, 1))
    
    # Visualize results
    visualization = visualize_results(df, analysis, history, predictions)
    visualization.show()
    
    # Print analysis results
    print("\nTime Series Analysis:")
    print("\nStationarity Test:")
    print(f"ADF Statistic: {analysis['stationarity']['test_statistic']:.4f}")
    print(f"p-value: {analysis['stationarity']['p_value']:.4f}")
    print("\nBasic Statistics:")
    print(analysis['statistics']) 