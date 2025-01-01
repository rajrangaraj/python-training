"""
Demonstration of advanced test doubles and dependency injection patterns in testing.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Protocol
import pytest
from unittest.mock import Mock, AsyncMock, patch
import asyncio
import json

# Domain Models
@dataclass
class WeatherData:
    """Weather data model."""
    temperature: float
    humidity: float
    pressure: float
    timestamp: datetime

@dataclass
class Forecast:
    """Weather forecast model."""
    location: str
    predictions: List[WeatherData]
    generated_at: datetime

# Interfaces/Protocols
class WeatherAPI(Protocol):
    """Protocol for weather API interactions."""
    
    async def get_current_weather(self, location: str) -> WeatherData:
        """Get current weather for location."""
        ...
    
    async def get_forecast(self, location: str, days: int) -> Forecast:
        """Get weather forecast for location."""
        ...

class CacheService(Protocol):
    """Protocol for caching service."""
    
    async def get(self, key: str) -> Optional[str]:
        """Get value from cache."""
        ...
    
    async def set(self, key: str, value: str, ttl: int) -> bool:
        """Set value in cache with TTL."""
        ...

class NotificationService(Protocol):
    """Protocol for notification service."""
    
    async def send_alert(self, message: str, severity: str) -> bool:
        """Send weather alert."""
        ...

# Implementations
class OpenWeatherAPI:
    """Real implementation of WeatherAPI using OpenWeather."""
    
    def __init__(self, api_key: str, base_url: str):
        self.api_key = api_key
        self.base_url = base_url
    
    async def get_current_weather(self, location: str) -> WeatherData:
        """Get current weather (real implementation would call API)."""
        raise NotImplementedError("Real API calls not implemented for demo")
    
    async def get_forecast(self, location: str, days: int) -> Forecast:
        """Get forecast (real implementation would call API)."""
        raise NotImplementedError("Real API calls not implemented for demo")

class RedisCache:
    """Real implementation of CacheService using Redis."""
    
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
    
    async def get(self, key: str) -> Optional[str]:
        """Get value from Redis (real implementation would use Redis)."""
        raise NotImplementedError("Real Redis calls not implemented for demo")
    
    async def set(self, key: str, value: str, ttl: int) -> bool:
        """Set value in Redis (real implementation would use Redis)."""
        raise NotImplementedError("Real Redis calls not implemented for demo")

# Service Layer
class WeatherService:
    """Weather service combining API, cache, and notifications."""
    
    def __init__(
        self,
        weather_api: WeatherAPI,
        cache: CacheService,
        notifications: NotificationService
    ):
        self.weather_api = weather_api
        self.cache = cache
        self.notifications = notifications
    
    async def get_weather(self, location: str) -> WeatherData:
        """Get weather data with caching."""
        cache_key = f"weather:{location}"
        
        # Try cache first
        if cached_data := await self.cache.get(cache_key):
            return WeatherData(**json.loads(cached_data))
        
        # Get from API
        weather = await self.weather_api.get_current_weather(location)
        
        # Cache the result
        await self.cache.set(
            cache_key,
            json.dumps({
                "temperature": weather.temperature,
                "humidity": weather.humidity,
                "pressure": weather.pressure,
                "timestamp": weather.timestamp.isoformat()
            }),
            ttl=300  # 5 minutes
        )
        
        return weather
    
    async def monitor_temperature(self, location: str, threshold: float) -> None:
        """Monitor temperature and send alerts if threshold exceeded."""
        weather = await self.get_weather(location)
        
        if weather.temperature > threshold:
            await self.notifications.send_alert(
                f"High temperature alert: {weather.temperature}°C in {location}",
                severity="warning"
            )

# Tests
class TestWeatherService:
    """Tests for WeatherService using various test doubles."""
    
    @pytest.fixture
    def mock_weather_api(self) -> Mock:
        """Provide a mock weather API."""
        api = AsyncMock(spec=WeatherAPI)
        api.get_current_weather.return_value = WeatherData(
            temperature=25.0,
            humidity=60.0,
            pressure=1013.0,
            timestamp=datetime.now()
        )
        return api
    
    @pytest.fixture
    def mock_cache(self) -> Mock:
        """Provide a mock cache service."""
        cache = AsyncMock(spec=CacheService)
        cache.get.return_value = None  # Default to cache miss
        cache.set.return_value = True
        return cache
    
    @pytest.fixture
    def mock_notifications(self) -> Mock:
        """Provide a mock notification service."""
        return AsyncMock(spec=NotificationService)
    
    @pytest.fixture
    def weather_service(
        self,
        mock_weather_api: Mock,
        mock_cache: Mock,
        mock_notifications: Mock
    ) -> WeatherService:
        """Provide a WeatherService with mock dependencies."""
        return WeatherService(mock_weather_api, mock_cache, mock_notifications)
    
    @pytest.mark.asyncio
    async def test_get_weather_cache_miss(self, weather_service, mock_weather_api, mock_cache):
        """Test weather retrieval with cache miss."""
        weather = await weather_service.get_weather("London")
        
        assert weather.temperature == 25.0
        mock_weather_api.get_current_weather.assert_called_once_with("London")
        mock_cache.set.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_weather_cache_hit(self, weather_service, mock_weather_api, mock_cache):
        """Test weather retrieval with cache hit."""
        # Setup cache hit
        cached_data = {
            "temperature": 20.0,
            "humidity": 55.0,
            "pressure": 1012.0,
            "timestamp": datetime.now().isoformat()
        }
        mock_cache.get.return_value = json.dumps(cached_data)
        
        weather = await weather_service.get_weather("London")
        
        assert weather.temperature == 20.0
        mock_weather_api.get_current_weather.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_temperature_alert(
        self,
        weather_service,
        mock_weather_api,
        mock_notifications
    ):
        """Test temperature monitoring with alert."""
        # Setup high temperature
        mock_weather_api.get_current_weather.return_value = WeatherData(
            temperature=35.0,
            humidity=60.0,
            pressure=1013.0,
            timestamp=datetime.now()
        )
        
        await weather_service.monitor_temperature("London", threshold=30.0)
        
        mock_notifications.send_alert.assert_called_once()
        assert "High temperature alert" in mock_notifications.send_alert.call_args[0][0]

if __name__ == '__main__':
    pytest.main([__file__]) 