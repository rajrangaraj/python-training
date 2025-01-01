"""
Demonstration of microservices architecture and API design patterns.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from datetime import datetime
import json
import uuid
import asyncio
import aiohttp
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
import redis
import jwt
from abc import ABC, abstractmethod

# Domain Models
@dataclass
class User:
    id: str
    username: str
    email: str
    created_at: datetime

@dataclass
class Order:
    id: str
    user_id: str
    items: List[Dict[str, Any]]
    total: float
    status: str
    created_at: datetime

# Service Interfaces
class UserService(ABC):
    @abstractmethod
    async def create_user(self, username: str, email: str) -> User:
        pass
    
    @abstractmethod
    async def get_user(self, user_id: str) -> Optional[User]:
        pass

class OrderService(ABC):
    @abstractmethod
    async def create_order(self, user_id: str, items: List[Dict[str, Any]]) -> Order:
        pass
    
    @abstractmethod
    async def get_order(self, order_id: str) -> Optional[Order]:
        pass

# API Models
class UserCreate(BaseModel):
    username: str
    email: str

class OrderCreate(BaseModel):
    items: List[Dict[str, Any]]

# Service Implementations
class UserServiceImpl(UserService):
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
    
    async def create_user(self, username: str, email: str) -> User:
        user = User(
            id=str(uuid.uuid4()),
            username=username,
            email=email,
            created_at=datetime.now()
        )
        
        await self.redis.set(
            f"user:{user.id}",
            json.dumps({
                'id': user.id,
                'username': user.username,
                'email': user.email,
                'created_at': user.created_at.isoformat()
            })
        )
        
        return user
    
    async def get_user(self, user_id: str) -> Optional[User]:
        user_data = await self.redis.get(f"user:{user_id}")
        if not user_data:
            return None
        
        data = json.loads(user_data)
        return User(
            id=data['id'],
            username=data['username'],
            email=data['email'],
            created_at=datetime.fromisoformat(data['created_at'])
        )

class OrderServiceImpl(OrderService):
    def __init__(self, redis_client: redis.Redis, user_service: UserService):
        self.redis = redis_client
        self.user_service = user_service
    
    async def create_order(self, user_id: str, items: List[Dict[str, Any]]) -> Order:
        # Verify user exists
        user = await self.user_service.get_user(user_id)
        if not user:
            raise ValueError("User not found")
        
        # Calculate total
        total = sum(item['price'] * item['quantity'] for item in items)
        
        order = Order(
            id=str(uuid.uuid4()),
            user_id=user_id,
            items=items,
            total=total,
            status="pending",
            created_at=datetime.now()
        )
        
        await self.redis.set(
            f"order:{order.id}",
            json.dumps({
                'id': order.id,
                'user_id': order.user_id,
                'items': order.items,
                'total': order.total,
                'status': order.status,
                'created_at': order.created_at.isoformat()
            })
        )
        
        # Publish event
        await self.redis.publish(
            'order_created',
            json.dumps({'order_id': order.id})
        )
        
        return order
    
    async def get_order(self, order_id: str) -> Optional[Order]:
        order_data = await self.redis.get(f"order:{order_id}")
        if not order_data:
            return None
        
        data = json.loads(order_data)
        return Order(
            id=data['id'],
            user_id=data['user_id'],
            items=data['items'],
            total=data['total'],
            status=data['status'],
            created_at=datetime.fromisoformat(data['created_at'])
        )

# API Gateway
app = FastAPI(title="E-commerce Microservices")

# Dependency Injection
def get_redis():
    return redis.Redis(host='localhost', port=6379, db=0)

def get_user_service(redis=Depends(get_redis)):
    return UserServiceImpl(redis)

def get_order_service(redis=Depends(get_redis), user_service=Depends(get_user_service)):
    return OrderServiceImpl(redis, user_service)

# Authentication Middleware
async def get_current_user(authorization: str = Depends()):
    try:
        payload = jwt.decode(authorization, "secret", algorithms=["HS256"])
        return payload['sub']
    except:
        raise HTTPException(status_code=401, detail="Invalid token")

# API Routes
@app.post("/users", response_model=Dict[str, Any])
async def create_user(
    user_data: UserCreate,
    user_service: UserService = Depends(get_user_service)
):
    try:
        user = await user_service.create_user(
            username=user_data.username,
            email=user_data.email
        )
        return {
            "id": user.id,
            "username": user.username,
            "email": user.email,
            "created_at": user.created_at.isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/orders", response_model=Dict[str, Any])
async def create_order(
    order_data: OrderCreate,
    current_user: str = Depends(get_current_user),
    order_service: OrderService = Depends(get_order_service)
):
    try:
        order = await order_service.create_order(
            user_id=current_user,
            items=order_data.items
        )
        return {
            "id": order.id,
            "user_id": order.user_id,
            "items": order.items,
            "total": order.total,
            "status": order.status,
            "created_at": order.created_at.isoformat()
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Event Handlers
async def handle_order_created(order_id: str):
    """Simulate order processing workflow."""
    await asyncio.sleep(1)  # Simulate processing time
    redis_client = get_redis()
    await redis_client.set(f"order:{order_id}:status", "processing")
    
    await asyncio.sleep(2)  # Simulate processing time
    await redis_client.set(f"order:{order_id}:status", "completed")

# Event Listeners
async def start_event_listeners():
    """Start background tasks for event handling."""
    redis_client = get_redis()
    pubsub = redis_client.pubsub()
    await pubsub.subscribe('order_created')
    
    async for message in pubsub.listen():
        if message['type'] == 'message':
            data = json.loads(message['data'])
            asyncio.create_task(handle_order_created(data['order_id']))

@app.on_event("startup")
async def startup_event():
    """Initialize services and start event listeners."""
    asyncio.create_task(start_event_listeners())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 