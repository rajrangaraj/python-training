"""
Demonstration of API Gateway and Service Mesh patterns.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import asyncio
import aiohttp
import jwt
from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer
import redis
from prometheus_client import Counter, Histogram
import structlog
import circuit_breaker
from ratelimit import RateLimiter
import opentracing

# Configure logging
logger = structlog.get_logger()

# Initialize FastAPI app
app = FastAPI(title="API Gateway")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Metrics
REQUEST_COUNT = Counter(
    'gateway_request_total',
    'Total requests through gateway',
    ['service', 'endpoint', 'method']
)

LATENCY = Histogram(
    'gateway_request_latency_seconds',
    'Request latency in seconds',
    ['service', 'endpoint']
)

# Service Registry
@dataclass
class ServiceEndpoint:
    name: str
    url: str
    health_check_url: str
    timeout: float = 5.0
    circuit_breaker: Optional[circuit_breaker.CircuitBreaker] = None
    rate_limiter: Optional[RateLimiter] = None

class ServiceRegistry:
    """Service registry for managing microservices."""
    
    def __init__(self):
        self.services: Dict[str, ServiceEndpoint] = {}
    
    def register_service(self, service: ServiceEndpoint):
        """Register a new service."""
        self.services[service.name] = service
        logger.info(
            "service_registered",
            service=service.name,
            url=service.url
        )
    
    def get_service(self, name: str) -> Optional[ServiceEndpoint]:
        """Get service by name."""
        return self.services.get(name)
    
    async def health_check(self) -> Dict[str, bool]:
        """Check health of all registered services."""
        results = {}
        async with aiohttp.ClientSession() as session:
            for name, service in self.services.items():
                try:
                    async with session.get(
                        service.health_check_url,
                        timeout=service.timeout
                    ) as response:
                        results[name] = response.status == 200
                except Exception:
                    results[name] = False
        return results

# API Gateway
class APIGateway:
    """API Gateway implementation."""
    
    def __init__(
        self,
        service_registry: ServiceRegistry,
        redis_client: redis.Redis
    ):
        self.service_registry = service_registry
        self.redis = redis_client
        self.tracer = opentracing.global_tracer()
    
    async def route_request(
        self,
        service_name: str,
        endpoint: str,
        method: str,
        headers: Dict[str, str],
        data: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Route request to appropriate service."""
        service = self.service_registry.get_service(service_name)
        if not service:
            raise HTTPException(
                status_code=404,
                detail=f"Service {service_name} not found"
            )
        
        # Start tracing span
        with self.tracer.start_span(
            f"{service_name}.{endpoint}",
            tags={
                "service": service_name,
                "endpoint": endpoint,
                "method": method
            }
        ) as span:
            try:
                # Rate limiting
                if service.rate_limiter and not service.rate_limiter.acquire():
                    raise HTTPException(
                        status_code=429,
                        detail="Too many requests"
                    )
                
                # Circuit breaker
                if service.circuit_breaker and not service.circuit_breaker.can_execute():
                    raise HTTPException(
                        status_code=503,
                        detail="Service temporarily unavailable"
                    )
                
                # Cache check for GET requests
                cache_key = f"{service_name}:{endpoint}:{method}"
                if method == "GET":
                    cached = await self.redis.get(cache_key)
                    if cached:
                        return json.loads(cached)
                
                # Make request
                start_time = time.time()
                async with aiohttp.ClientSession() as session:
                    url = f"{service.url}{endpoint}"
                    async with session.request(
                        method,
                        url,
                        headers=headers,
                        json=data,
                        timeout=service.timeout
                    ) as response:
                        # Record metrics
                        REQUEST_COUNT.labels(
                            service_name,
                            endpoint,
                            method
                        ).inc()
                        
                        LATENCY.labels(
                            service_name,
                            endpoint
                        ).observe(time.time() - start_time)
                        
                        # Process response
                        if response.status >= 400:
                            if service.circuit_breaker:
                                service.circuit_breaker.record_failure()
                            raise HTTPException(
                                status_code=response.status,
                                detail=await response.text()
                            )
                        
                        result = await response.json()
                        
                        # Cache successful GET responses
                        if method == "GET":
                            await self.redis.setex(
                                cache_key,
                                300,  # 5 minutes TTL
                                json.dumps(result)
                            )
                        
                        if service.circuit_breaker:
                            service.circuit_breaker.record_success()
                        
                        return result
            
            except Exception as e:
                span.set_tag("error", True)
                span.log_kv({"event": "error", "message": str(e)})
                raise

# Authentication middleware
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

async def verify_token(
    authorization: str = Depends(oauth2_scheme)
) -> Dict[str, Any]:
    """Verify JWT token."""
    try:
        return jwt.decode(
            authorization,
            "secret",
            algorithms=["HS256"]
        )
    except jwt.PyJWTError:
        raise HTTPException(
            status_code=401,
            detail="Invalid authentication credentials"
        )

# Service mesh sidecar
class ServiceMeshSidecar:
    """Service mesh sidecar implementation."""
    
    def __init__(self, service_name: str):
        self.service_name = service_name
        self.tracer = opentracing.global_tracer()
    
    async def handle_ingress(
        self,
        request: Dict[str, Any],
        headers: Dict[str, str]
    ) -> Any:
        """Handle incoming requests."""
        with self.tracer.start_span(
            f"{self.service_name}.ingress",
            tags={"service": self.service_name}
        ) as span:
            # Add tracing headers
            headers.update(self.tracer.inject(span, format="http_headers"))
            
            # Handle request
            try:
                # Process request...
                return {"status": "success"}
            except Exception as e:
                span.set_tag("error", True)
                span.log_kv({"event": "error", "message": str(e)})
                raise
    
    async def handle_egress(
        self,
        request: Dict[str, Any],
        target_service: str
    ) -> Any:
        """Handle outgoing requests."""
        with self.tracer.start_span(
            f"{self.service_name}.egress",
            tags={
                "service": self.service_name,
                "target": target_service
            }
        ) as span:
            try:
                # Process request...
                return {"status": "success"}
            except Exception as e:
                span.set_tag("error", True)
                span.log_kv({"event": "error", "message": str(e)})
                raise

# API Routes
@app.get("/services/{service_name}/{endpoint}")
async def proxy_get_request(
    service_name: str,
    endpoint: str,
    token: Dict[str, Any] = Depends(verify_token),
    headers: Dict[str, str] = Header(None)
):
    """Proxy GET request to service."""
    return await gateway.route_request(
        service_name,
        endpoint,
        "GET",
        headers
    )

@app.post("/services/{service_name}/{endpoint}")
async def proxy_post_request(
    service_name: str,
    endpoint: str,
    data: Dict[str, Any],
    token: Dict[str, Any] = Depends(verify_token),
    headers: Dict[str, str] = Header(None)
):
    """Proxy POST request to service."""
    return await gateway.route_request(
        service_name,
        endpoint,
        "POST",
        headers,
        data
    )

@app.get("/health")
async def health_check():
    """Gateway health check endpoint."""
    service_health = await service_registry.health_check()
    return {
        "gateway": "healthy",
        "services": service_health
    }

if __name__ == "__main__":
    # Initialize components
    service_registry = ServiceRegistry()
    redis_client = redis.Redis(host='localhost', port=6379, db=0)
    gateway = APIGateway(service_registry, redis_client)
    
    # Register services
    service_registry.register_service(
        ServiceEndpoint(
            name="users",
            url="http://users-service:8000",
            health_check_url="http://users-service:8000/health",
            circuit_breaker=circuit_breaker.CircuitBreaker(),
            rate_limiter=RateLimiter(max_calls=100, period=60)
        )
    )
    
    # Start server
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 