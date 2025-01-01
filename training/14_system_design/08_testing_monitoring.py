"""
Demonstration of testing and monitoring patterns for system design.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import time
import json
import logging
import asyncio
import aiohttp
from abc import ABC, abstractmethod
import prometheus_client as prom
from opentelemetry import trace, metrics
from opentelemetry.trace import Status, StatusCode
import structlog
from unittest.mock import Mock, patch
import pytest
import asynctest

# Configure structured logging
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
    ]
)
logger = structlog.get_logger()

# Metrics
REQUEST_LATENCY = prom.Histogram(
    'request_latency_seconds',
    'Request latency in seconds',
    ['method', 'endpoint']
)

REQUEST_COUNT = prom.Counter(
    'request_count_total',
    'Total request count',
    ['method', 'endpoint', 'status']
)

ERROR_COUNT = prom.Counter(
    'error_count_total',
    'Total error count',
    ['type']
)

# Tracing
tracer = trace.get_tracer(__name__)

# Domain Models
@dataclass
class Order:
    id: str
    user_id: str
    items: List[Dict[str, Any]]
    total: float
    status: str
    created_at: datetime

# Service Interfaces
class OrderService(ABC):
    @abstractmethod
    async def create_order(self, user_id: str, items: List[Dict[str, Any]]) -> Order:
        pass
    
    @abstractmethod
    async def get_order(self, order_id: str) -> Optional[Order]:
        pass

# Instrumented Service Implementation
class InstrumentedOrderService(OrderService):
    """Order service with instrumentation."""
    
    def __init__(self):
        self.orders: Dict[str, Order] = {}
    
    async def create_order(
        self,
        user_id: str,
        items: List[Dict[str, Any]]
    ) -> Order:
        with tracer.start_as_current_span("create_order") as span:
            try:
                start_time = time.time()
                
                # Business logic
                order = Order(
                    id=str(len(self.orders) + 1),
                    user_id=user_id,
                    items=items,
                    total=sum(item["price"] * item["quantity"] for item in items),
                    status="pending",
                    created_at=datetime.utcnow()
                )
                
                # Simulate processing
                await asyncio.sleep(0.1)
                
                self.orders[order.id] = order
                
                # Record metrics
                REQUEST_LATENCY.labels(
                    method='create_order',
                    endpoint='/orders'
                ).observe(time.time() - start_time)
                
                REQUEST_COUNT.labels(
                    method='create_order',
                    endpoint='/orders',
                    status='success'
                ).inc()
                
                # Structured logging
                logger.info(
                    "order_created",
                    order_id=order.id,
                    user_id=user_id,
                    total=order.total
                )
                
                # Add span attributes
                span.set_attribute("order.id", order.id)
                span.set_attribute("order.total", order.total)
                
                return order
            
            except Exception as e:
                # Record error metrics
                ERROR_COUNT.labels(type=type(e).__name__).inc()
                
                # Error logging
                logger.error(
                    "order_creation_failed",
                    error=str(e),
                    user_id=user_id
                )
                
                # Update span with error
                span.set_status(Status(StatusCode.ERROR))
                span.record_exception(e)
                raise
    
    async def get_order(self, order_id: str) -> Optional[Order]:
        with tracer.start_as_current_span("get_order") as span:
            try:
                start_time = time.time()
                
                # Business logic
                order = self.orders.get(order_id)
                
                # Record metrics
                REQUEST_LATENCY.labels(
                    method='get_order',
                    endpoint='/orders/{id}'
                ).observe(time.time() - start_time)
                
                REQUEST_COUNT.labels(
                    method='get_order',
                    endpoint='/orders/{id}',
                    status='success' if order else 'not_found'
                ).inc()
                
                # Logging
                if order:
                    logger.info(
                        "order_retrieved",
                        order_id=order_id
                    )
                else:
                    logger.warning(
                        "order_not_found",
                        order_id=order_id
                    )
                
                # Add span attributes
                span.set_attribute("order.id", order_id)
                span.set_attribute("order.found", bool(order))
                
                return order
            
            except Exception as e:
                ERROR_COUNT.labels(type=type(e).__name__).inc()
                logger.error(
                    "order_retrieval_failed",
                    error=str(e),
                    order_id=order_id
                )
                span.set_status(Status(StatusCode.ERROR))
                span.record_exception(e)
                raise

# Health Check
async def health_check() -> Dict[str, Any]:
    """System health check implementation."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "components": {
            "database": "up",
            "cache": "up",
            "message_queue": "up"
        },
        "metrics": {
            "request_count": REQUEST_COUNT._value.sum(),
            "error_count": ERROR_COUNT._value.sum()
        }
    }

# Tests
class TestOrderService:
    @pytest.fixture
    def service(self):
        return InstrumentedOrderService()
    
    @pytest.mark.asyncio
    async def test_create_order(self, service):
        # Arrange
        user_id = "user123"
        items = [
            {"product_id": "prod1", "quantity": 2, "price": 10.0},
            {"product_id": "prod2", "quantity": 1, "price": 20.0}
        ]
        
        # Act
        order = await service.create_order(user_id, items)
        
        # Assert
        assert order.user_id == user_id
        assert order.total == 40.0
        assert order.status == "pending"
        assert len(order.items) == 2
    
    @pytest.mark.asyncio
    async def test_get_order(self, service):
        # Arrange
        order = await service.create_order(
            "user123",
            [{"product_id": "prod1", "quantity": 1, "price": 10.0}]
        )
        
        # Act
        retrieved_order = await service.get_order(order.id)
        
        # Assert
        assert retrieved_order is not None
        assert retrieved_order.id == order.id
        assert retrieved_order.total == 10.0
    
    @pytest.mark.asyncio
    async def test_get_nonexistent_order(self, service):
        # Act
        order = await service.get_order("nonexistent")
        
        # Assert
        assert order is None

# Load Testing
async def load_test(
    num_requests: int,
    concurrency: int
):
    """Simple load testing implementation."""
    async def make_request():
        service = InstrumentedOrderService()
        items = [{"product_id": "test", "quantity": 1, "price": 10.0}]
        
        start_time = time.time()
        try:
            await service.create_order("test_user", items)
            return time.time() - start_time, True
        except Exception:
            return time.time() - start_time, False
    
    tasks = [make_request() for _ in range(num_requests)]
    results = await asyncio.gather(*tasks)
    
    latencies = [r[0] for r in results]
    success_rate = sum(1 for r in results if r[1]) / len(results)
    
    return {
        "total_requests": num_requests,
        "success_rate": success_rate,
        "avg_latency": sum(latencies) / len(latencies),
        "min_latency": min(latencies),
        "max_latency": max(latencies)
    }

async def demonstrate_testing_monitoring():
    """Demonstrate testing and monitoring patterns."""
    
    # Initialize service
    service = InstrumentedOrderService()
    
    # Create some orders
    for i in range(5):
        try:
            await service.create_order(
                f"user{i}",
                [{"product_id": f"prod{i}", "quantity": 1, "price": 10.0}]
            )
        except Exception as e:
            logger.error("demo_error", error=str(e))
    
    # Retrieve orders
    for i in range(6):  # One extra to test non-existent order
        try:
            await service.get_order(str(i))
        except Exception as e:
            logger.error("demo_error", error=str(e))
    
    # Run health check
    health_status = await health_check()
    logger.info("health_check", **health_status)
    
    # Run load test
    load_test_results = await load_test(
        num_requests=100,
        concurrency=10
    )
    logger.info("load_test_completed", **load_test_results)

if __name__ == "__main__":
    # Start Prometheus HTTP server
    prom.start_http_server(8000)
    
    # Run demonstration
    asyncio.run(demonstrate_testing_monitoring()) 