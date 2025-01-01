"""
Demonstration of Flask microservices architecture and service discovery.
"""

from flask import Flask, jsonify, request
from flask_restx import Api, Resource, fields
import consul
import requests
import json
import os
from datetime import datetime
import socket
import random
from functools import wraps
import jwt
from circuitbreaker import circuit
import prometheus_client
from prometheus_client import Counter, Histogram
import logging
from opentelemetry import trace
from opentelemetry.exporter import jaeger
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

app = Flask(__name__)
api = Api(app)

# Configuration
SERVICE_NAME = os.getenv('SERVICE_NAME', 'user-service')
SERVICE_HOST = os.getenv('SERVICE_HOST', 'localhost')
SERVICE_PORT = int(os.getenv('SERVICE_PORT', 5000))
CONSUL_HOST = os.getenv('CONSUL_HOST', 'localhost')
CONSUL_PORT = int(os.getenv('CONSUL_PORT', 8500))

# Initialize Consul client
consul_client = consul.Consul(host=CONSUL_HOST, port=CONSUL_PORT)

# Initialize tracing
trace.set_tracer_provider(TracerProvider())
jaeger_exporter = jaeger.JaegerSpanExporter(
    agent_host_name="localhost",
    agent_port=6831,
)
trace.get_tracer_provider().add_span_processor(
    BatchSpanProcessor(jaeger_exporter)
)
tracer = trace.get_tracer(__name__)

# Metrics
REQUEST_COUNT = Counter(
    'request_count', 'App Request Count',
    ['app_name', 'method', 'endpoint', 'http_status']
)
REQUEST_LATENCY = Histogram(
    'request_latency_seconds', 'Request latency',
    ['app_name', 'endpoint']
)

# Service registry
class ServiceRegistry:
    def __init__(self):
        self.consul = consul_client
        self.service_id = f"{SERVICE_NAME}-{socket.gethostname()}"
    
    def register(self):
        """Register service with Consul."""
        self.consul.agent.service.register(
            name=SERVICE_NAME,
            service_id=self.service_id,
            address=SERVICE_HOST,
            port=SERVICE_PORT,
            tags=['flask', 'microservice'],
            check={
                'http': f'http://{SERVICE_HOST}:{SERVICE_PORT}/health',
                'interval': '10s',
                'timeout': '5s'
            }
        )
        logging.info(f"Registered service: {self.service_id}")
    
    def deregister(self):
        """Deregister service from Consul."""
        self.consul.agent.service.deregister(self.service_id)
        logging.info(f"Deregistered service: {self.service_id}")
    
    def get_service(self, service_name):
        """Get service details from Consul."""
        _, services = self.consul.health.service(service_name, passing=True)
        if services:
            service = random.choice(services)
            return {
                'id': service['Service']['ID'],
                'address': service['Service']['Address'],
                'port': service['Service']['Port']
            }
        return None

# Circuit breaker configuration
class CircuitBreakerConfig:
    FAILURE_THRESHOLD = 5
    RECOVERY_TIMEOUT = 60
    EXPECTED_EXCEPTION = (requests.exceptions.RequestException,)

# Service client with circuit breaker
@circuit(failure_threshold=CircuitBreakerConfig.FAILURE_THRESHOLD,
        recovery_timeout=CircuitBreakerConfig.RECOVERY_TIMEOUT,
        expected_exception=CircuitBreakerConfig.EXPECTED_EXCEPTION)
def service_call(url, method='GET', **kwargs):
    """Make service call with circuit breaker pattern."""
    response = requests.request(method, url, **kwargs)
    response.raise_for_status()
    return response.json()

# Middleware
def metrics_middleware():
    """Middleware to collect metrics."""
    def decorator(f):
        @wraps(f)
        def wrapped(*args, **kwargs):
            start_time = datetime.utcnow()
            
            response = f(*args, **kwargs)
            
            status_code = response[1] if isinstance(response, tuple) else 200
            REQUEST_COUNT.labels(
                app_name=SERVICE_NAME,
                method=request.method,
                endpoint=request.endpoint,
                http_status=status_code
            ).inc()
            
            latency = (datetime.utcnow() - start_time).total_seconds()
            REQUEST_LATENCY.labels(
                app_name=SERVICE_NAME,
                endpoint=request.endpoint
            ).observe(latency)
            
            return response
        return wrapped
    return decorator

# API models
service_model = api.model('Service', {
    'name': fields.String(required=True),
    'address': fields.String(required=True),
    'port': fields.Integer(required=True),
    'status': fields.String()
})

# API endpoints
@api.route('/health')
class Health(Resource):
    def get(self):
        """Health check endpoint."""
        return {'status': 'healthy'}, 200

@api.route('/metrics')
class Metrics(Resource):
    def get(self):
        """Prometheus metrics endpoint."""
        return Response(prometheus_client.generate_latest(),
                      mimetype='text/plain')

@api.route('/service/<service_name>')
class ServiceDiscovery(Resource):
    @metrics_middleware()
    def get(self, service_name):
        """Get service details."""
        with tracer.start_as_current_span("get_service") as span:
            span.set_attribute("service.name", service_name)
            service = ServiceRegistry().get_service(service_name)
            
            if not service:
                return {'error': 'Service not found'}, 404
            
            return service

@api.route('/call/<service_name>/<path:endpoint>')
class ServiceCall(Resource):
    @metrics_middleware()
    def get(self, service_name, endpoint):
        """Make service call."""
        with tracer.start_as_current_span("service_call") as span:
            span.set_attribute("service.name", service_name)
            span.set_attribute("endpoint", endpoint)
            
            service = ServiceRegistry().get_service(service_name)
            if not service:
                return {'error': 'Service not found'}, 404
            
            url = f"http://{service['address']}:{service['port']}/{endpoint}"
            try:
                return service_call(url)
            except Exception as e:
                return {'error': str(e)}, 500

# Service lifecycle management
service_registry = ServiceRegistry()

@app.before_first_request
def startup():
    """Register service on startup."""
    service_registry.register()

@app.teardown_appcontext
def shutdown(exception=None):
    """Deregister service on shutdown."""
    service_registry.deregister()

if __name__ == '__main__':
    app.run(host=SERVICE_HOST, port=SERVICE_PORT) 