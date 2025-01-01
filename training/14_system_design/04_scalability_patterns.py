"""
Demonstration of system scalability and performance patterns.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import time
import random
import asyncio
import aiohttp
import redis
from functools import lru_cache
from datetime import datetime, timedelta
import threading
from queue import Queue
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cache Patterns
class Cache:
    """Simple cache implementation with TTL support."""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
    
    async def get(self, key: str) -> Optional[str]:
        return await self.redis.get(key)
    
    async def set(self, key: str, value: str, ttl_seconds: int = 300):
        await self.redis.setex(key, ttl_seconds, value)
    
    async def delete(self, key: str):
        await self.redis.delete(key)

@lru_cache(maxsize=1000)
def compute_expensive_result(input_data: str) -> str:
    """Simulate expensive computation with in-memory caching."""
    time.sleep(1)  # Simulate processing
    return f"Result for {input_data}"

# Circuit Breaker Pattern
class CircuitBreaker:
    """Implement circuit breaker pattern for fault tolerance."""
    
    def __init__(self, failure_threshold: int = 5, reset_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.failures = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half-open
        self._lock = threading.Lock()
    
    def can_execute(self) -> bool:
        with self._lock:
            if self.state == "closed":
                return True
            
            if self.state == "open":
                if (datetime.now() - self.last_failure_time).seconds >= self.reset_timeout:
                    self.state = "half-open"
                    return True
                return False
            
            return True  # half-open state
    
    def record_success(self):
        with self._lock:
            self.failures = 0
            self.state = "closed"
    
    def record_failure(self):
        with self._lock:
            self.failures += 1
            self.last_failure_time = datetime.now()
            
            if self.failures >= self.failure_threshold:
                self.state = "open"

# Rate Limiting
class RateLimiter:
    """Token bucket rate limiter implementation."""
    
    def __init__(self, capacity: int, refill_rate: float):
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = capacity
        self.last_refill = time.time()
        self._lock = threading.Lock()
    
    def _refill(self):
        now = time.time()
        elapsed = now - self.last_refill
        tokens_to_add = elapsed * self.refill_rate
        
        self.tokens = min(self.capacity, self.tokens + tokens_to_add)
        self.last_refill = now
    
    def acquire(self) -> bool:
        with self._lock:
            self._refill()
            
            if self.tokens >= 1:
                self.tokens -= 1
                return True
            return False

# Load Balancer
class LoadBalancer:
    """Simple round-robin load balancer."""
    
    def __init__(self, backends: List[str]):
        self.backends = backends
        self.current = 0
        self._lock = threading.Lock()
    
    def get_next_backend(self) -> str:
        with self._lock:
            backend = self.backends[self.current]
            self.current = (self.current + 1) % len(self.backends)
            return backend

# Message Queue
class MessageQueue:
    """Simple message queue implementation."""
    
    def __init__(self):
        self.queues: Dict[str, Queue] = {}
        self._lock = threading.Lock()
    
    def create_queue(self, queue_name: str):
        with self._lock:
            if queue_name not in self.queues:
                self.queues[queue_name] = Queue()
    
    def publish(self, queue_name: str, message: Any):
        if queue_name in self.queues:
            self.queues[queue_name].put(message)
    
    def subscribe(self, queue_name: str) -> Optional[Any]:
        if queue_name in self.queues:
            return self.queues[queue_name].get()
        return None

# Example Service Using Patterns
class ScalableService:
    """Example service implementing scalability patterns."""
    
    def __init__(
        self,
        cache: Cache,
        circuit_breaker: CircuitBreaker,
        rate_limiter: RateLimiter,
        load_balancer: LoadBalancer,
        message_queue: MessageQueue
    ):
        self.cache = cache
        self.circuit_breaker = circuit_breaker
        self.rate_limiter = rate_limiter
        self.load_balancer = load_balancer
        self.message_queue = message_queue
    
    async def process_request(self, request_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
        # Rate limiting
        if not self.rate_limiter.acquire():
            raise Exception("Rate limit exceeded")
        
        # Check cache
        cache_key = f"request:{request_id}"
        cached_result = await self.cache.get(cache_key)
        if cached_result:
            return {"result": cached_result, "source": "cache"}
        
        # Circuit breaker
        if not self.circuit_breaker.can_execute():
            raise Exception("Service unavailable")
        
        try:
            # Get backend server
            backend = self.load_balancer.get_next_backend()
            
            # Process request
            result = await self._process_on_backend(backend, data)
            
            # Cache result
            await self.cache.set(cache_key, result)
            
            # Record success
            self.circuit_breaker.record_success()
            
            # Publish event
            self.message_queue.publish(
                "processed_requests",
                {"request_id": request_id, "result": result}
            )
            
            return {"result": result, "source": "backend"}
        
        except Exception as e:
            self.circuit_breaker.record_failure()
            raise
    
    async def _process_on_backend(self, backend: str, data: Dict[str, Any]) -> str:
        """Simulate processing request on backend server."""
        async with aiohttp.ClientSession() as session:
            async with session.post(backend, json=data) as response:
                if response.status == 200:
                    return await response.text()
                raise Exception(f"Backend error: {response.status}")

async def demonstrate_patterns():
    """Demonstrate scalability patterns."""
    
    # Initialize components
    redis_client = redis.Redis(host='localhost', port=6379, db=0)
    cache = Cache(redis_client)
    circuit_breaker = CircuitBreaker()
    rate_limiter = RateLimiter(capacity=100, refill_rate=10)
    load_balancer = LoadBalancer([
        "http://backend1:8080",
        "http://backend2:8080",
        "http://backend3:8080"
    ])
    message_queue = MessageQueue()
    
    service = ScalableService(
        cache=cache,
        circuit_breaker=circuit_breaker,
        rate_limiter=rate_limiter,
        load_balancer=load_balancer,
        message_queue=message_queue
    )
    
    # Process requests
    for i in range(5):
        request_id = f"req_{i}"
        try:
            result = await service.process_request(
                request_id,
                {"data": f"request_{i}"}
            )
            logger.info(f"Request {request_id}: {result}")
        except Exception as e:
            logger.error(f"Request {request_id} failed: {str(e)}")
        
        # Simulate varying load
        await asyncio.sleep(random.uniform(0.1, 0.5))

if __name__ == "__main__":
    asyncio.run(demonstrate_patterns()) 