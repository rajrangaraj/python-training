"""
Demonstration of database design patterns and data modeling concepts.
"""

from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime
import json
from abc import ABC, abstractmethod
from enum import Enum
import asyncio
import asyncpg
from sqlalchemy import create_engine, Column, Integer, String, ForeignKey, DateTime, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
from sqlalchemy.dialects.postgresql import JSONB
from pymongo import MongoClient
from redis import Redis

# Base Models
Base = declarative_base()

class UserRole(Enum):
    ADMIN = "admin"
    USER = "user"
    GUEST = "guest"

# Relational Models (SQL)
class User(Base):
    """User model with normalized relationships."""
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True)
    username = Column(String, unique=True, nullable=False)
    email = Column(String, unique=True, nullable=False)
    role = Column(String, nullable=False, default=UserRole.USER.value)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    
    # Relationships
    profile = relationship("UserProfile", back_populates="user", uselist=False)
    orders = relationship("Order", back_populates="user")
    addresses = relationship("Address", back_populates="user")

class UserProfile(Base):
    """One-to-One relationship example."""
    __tablename__ = 'user_profiles'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), unique=True)
    full_name = Column(String)
    phone = Column(String)
    bio = Column(String)
    
    # Relationship
    user = relationship("User", back_populates="profile")

class Order(Base):
    """Many-to-One relationship example."""
    __tablename__ = 'orders'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'))
    status = Column(String, nullable=False)
    total = Column(Integer, nullable=False)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="orders")
    items = relationship("OrderItem", back_populates="order")

class OrderItem(Base):
    """Many-to-One relationship example."""
    __tablename__ = 'order_items'
    
    id = Column(Integer, primary_key=True)
    order_id = Column(Integer, ForeignKey('orders.id'))
    product_id = Column(Integer, ForeignKey('products.id'))
    quantity = Column(Integer, nullable=False)
    price = Column(Integer, nullable=False)
    
    # Relationships
    order = relationship("Order", back_populates="items")
    product = relationship("Product")

class Product(Base):
    """Product model with JSON data."""
    __tablename__ = 'products'
    
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    price = Column(Integer, nullable=False)
    metadata = Column(JSONB)  # PostgreSQL-specific JSON type

class Address(Base):
    """Many-to-One relationship example."""
    __tablename__ = 'addresses'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'))
    type = Column(String, nullable=False)  # shipping, billing
    street = Column(String, nullable=False)
    city = Column(String, nullable=False)
    country = Column(String, nullable=False)
    
    # Relationship
    user = relationship("User", back_populates="addresses")

# NoSQL Models (MongoDB)
@dataclass
class UserDocument:
    """Denormalized document model for MongoDB."""
    username: str
    email: str
    role: str
    profile: Dict[str, Any]
    addresses: List[Dict[str, Any]]
    created_at: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "username": self.username,
            "email": self.email,
            "role": self.role,
            "profile": self.profile,
            "addresses": self.addresses,
            "created_at": self.created_at
        }

@dataclass
class OrderDocument:
    """Denormalized document model for MongoDB."""
    user_id: str
    items: List[Dict[str, Any]]
    total: float
    status: str
    shipping_address: Dict[str, Any]
    created_at: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "items": self.items,
            "total": self.total,
            "status": self.status,
            "shipping_address": self.shipping_address,
            "created_at": self.created_at
        }

# Database Access Patterns
class DatabaseRepository(ABC):
    """Abstract base class for database repositories."""
    
    @abstractmethod
    async def get_user(self, user_id: Union[int, str]) -> Optional[Dict[str, Any]]:
        pass
    
    @abstractmethod
    async def create_user(self, user_data: Dict[str, Any]) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    async def get_user_orders(self, user_id: Union[int, str]) -> List[Dict[str, Any]]:
        pass

class SQLRepository(DatabaseRepository):
    """SQL database repository implementation."""
    
    def __init__(self, connection_string: str):
        self.engine = create_engine(connection_string)
        self.Session = sessionmaker(bind=self.engine)
    
    async def get_user(self, user_id: int) -> Optional[Dict[str, Any]]:
        session = self.Session()
        try:
            user = session.query(User).filter(User.id == user_id).first()
            if not user:
                return None
            
            return {
                "id": user.id,
                "username": user.username,
                "email": user.email,
                "role": user.role,
                "profile": {
                    "full_name": user.profile.full_name,
                    "phone": user.profile.phone,
                    "bio": user.profile.bio
                } if user.profile else None
            }
        finally:
            session.close()
    
    async def create_user(self, user_data: Dict[str, Any]) -> Dict[str, Any]:
        session = self.Session()
        try:
            user = User(
                username=user_data["username"],
                email=user_data["email"],
                role=user_data.get("role", UserRole.USER.value)
            )
            
            if "profile" in user_data:
                user.profile = UserProfile(**user_data["profile"])
            
            session.add(user)
            session.commit()
            
            return await self.get_user(user.id)
        finally:
            session.close()
    
    async def get_user_orders(self, user_id: int) -> List[Dict[str, Any]]:
        session = self.Session()
        try:
            orders = session.query(Order).filter(Order.user_id == user_id).all()
            return [{
                "id": order.id,
                "total": order.total,
                "status": order.status,
                "items": [{
                    "product_id": item.product_id,
                    "quantity": item.quantity,
                    "price": item.price
                } for item in order.items],
                "created_at": order.created_at
            } for order in orders]
        finally:
            session.close()

class MongoRepository(DatabaseRepository):
    """MongoDB repository implementation."""
    
    def __init__(self, connection_string: str, database: str):
        self.client = MongoClient(connection_string)
        self.db = self.client[database]
    
    async def get_user(self, user_id: str) -> Optional[Dict[str, Any]]:
        user = self.db.users.find_one({"_id": user_id})
        return user if user else None
    
    async def create_user(self, user_data: Dict[str, Any]) -> Dict[str, Any]:
        user = UserDocument(
            username=user_data["username"],
            email=user_data["email"],
            role=user_data.get("role", UserRole.USER.value),
            profile=user_data.get("profile", {}),
            addresses=user_data.get("addresses", []),
            created_at=datetime.utcnow()
        )
        
        result = self.db.users.insert_one(user.to_dict())
        return await self.get_user(result.inserted_id)
    
    async def get_user_orders(self, user_id: str) -> List[Dict[str, Any]]:
        return list(self.db.orders.find({"user_id": user_id}))

# Cache Patterns
class CacheLayer:
    """Cache layer implementation."""
    
    def __init__(self, redis_client: Redis):
        self.redis = redis_client
    
    async def get_cached_user(self, user_id: Union[int, str]) -> Optional[Dict[str, Any]]:
        data = await self.redis.get(f"user:{user_id}")
        return json.loads(data) if data else None
    
    async def cache_user(self, user_id: Union[int, str], user_data: Dict[str, Any], ttl: int = 3600):
        await self.redis.setex(
            f"user:{user_id}",
            ttl,
            json.dumps(user_data)
        )
    
    async def invalidate_user(self, user_id: Union[int, str]):
        await self.redis.delete(f"user:{user_id}")

# Service Layer
class UserService:
    """Service layer implementing caching and database access."""
    
    def __init__(
        self,
        db_repository: DatabaseRepository,
        cache_layer: CacheLayer
    ):
        self.db = db_repository
        self.cache = cache_layer
    
    async def get_user(self, user_id: Union[int, str]) -> Optional[Dict[str, Any]]:
        # Try cache first
        user = await self.cache.get_cached_user(user_id)
        if user:
            return user
        
        # Get from database
        user = await self.db.get_user(user_id)
        if user:
            # Cache for future requests
            await self.cache.cache_user(user_id, user)
        
        return user
    
    async def create_user(self, user_data: Dict[str, Any]) -> Dict[str, Any]:
        user = await self.db.create_user(user_data)
        await self.cache.cache_user(user["id"], user)
        return user
    
    async def get_user_orders(self, user_id: Union[int, str]) -> List[Dict[str, Any]]:
        return await self.db.get_user_orders(user_id)

async def demonstrate_database_patterns():
    """Demonstrate database patterns."""
    
    # Initialize components
    sql_repo = SQLRepository("postgresql://user:pass@localhost/dbname")
    mongo_repo = MongoRepository("mongodb://localhost:27017", "dbname")
    cache_layer = CacheLayer(Redis(host='localhost', port=6379, db=0))
    
    # Create services
    sql_service = UserService(sql_repo, cache_layer)
    mongo_service = UserService(mongo_repo, cache_layer)
    
    # Demonstrate usage
    user_data = {
        "username": "john_doe",
        "email": "john@example.com",
        "profile": {
            "full_name": "John Doe",
            "phone": "1234567890",
            "bio": "Software Engineer"
        }
    }
    
    # Create user
    sql_user = await sql_service.create_user(user_data)
    mongo_user = await mongo_service.create_user(user_data)
    
    # Get user (should hit cache)
    cached_sql_user = await sql_service.get_user(sql_user["id"])
    cached_mongo_user = await mongo_service.get_user(mongo_user["id"])
    
    # Get orders
    sql_orders = await sql_service.get_user_orders(sql_user["id"])
    mongo_orders = await mongo_service.get_user_orders(mongo_user["id"])

if __name__ == "__main__":
    asyncio.run(demonstrate_database_patterns()) 