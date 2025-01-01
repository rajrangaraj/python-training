"""
Demonstration of security patterns and authentication systems.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import jwt
import bcrypt
import secrets
from abc import ABC, abstractmethod
import logging
import asyncio
import redis
from fastapi import FastAPI, Depends, HTTPException, Security
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel, EmailStr

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Domain Models
@dataclass
class User:
    id: str
    username: str
    email: str
    hashed_password: str
    roles: List[str]
    is_active: bool
    created_at: datetime

@dataclass
class Token:
    access_token: str
    refresh_token: str
    token_type: str = "bearer"

# Security Interfaces
class PasswordHasher(ABC):
    @abstractmethod
    def hash_password(self, password: str) -> str:
        pass
    
    @abstractmethod
    def verify_password(self, password: str, hashed_password: str) -> bool:
        pass

class TokenProvider(ABC):
    @abstractmethod
    def create_access_token(self, data: Dict[str, Any], expires_delta: timedelta) -> str:
        pass
    
    @abstractmethod
    def create_refresh_token(self, data: Dict[str, Any]) -> str:
        pass
    
    @abstractmethod
    def verify_token(self, token: str) -> Dict[str, Any]:
        pass

# Security Implementations
class BcryptPasswordHasher(PasswordHasher):
    """Bcrypt password hashing implementation."""
    
    def hash_password(self, password: str) -> str:
        salt = bcrypt.gensalt()
        return bcrypt.hashpw(password.encode(), salt).decode()
    
    def verify_password(self, password: str, hashed_password: str) -> bool:
        return bcrypt.checkpw(
            password.encode(),
            hashed_password.encode()
        )

class JWTTokenProvider(TokenProvider):
    """JWT token provider implementation."""
    
    def __init__(self, secret_key: str, algorithm: str = "HS256"):
        self.secret_key = secret_key
        self.algorithm = algorithm
    
    def create_access_token(
        self,
        data: Dict[str, Any],
        expires_delta: timedelta
    ) -> str:
        to_encode = data.copy()
        expire = datetime.utcnow() + expires_delta
        to_encode.update({"exp": expire})
        return jwt.encode(
            to_encode,
            self.secret_key,
            algorithm=self.algorithm
        )
    
    def create_refresh_token(self, data: Dict[str, Any]) -> str:
        to_encode = data.copy()
        to_encode.update({
            "exp": datetime.utcnow() + timedelta(days=30),
            "refresh": True
        })
        return jwt.encode(
            to_encode,
            self.secret_key,
            algorithm=self.algorithm
        )
    
    def verify_token(self, token: str) -> Dict[str, Any]:
        try:
            return jwt.decode(
                token,
                self.secret_key,
                algorithms=[self.algorithm]
            )
        except jwt.PyJWTError:
            raise ValueError("Invalid token")

# Rate Limiting
class RateLimiter:
    """Rate limiting using Redis."""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
    
    async def is_rate_limited(
        self,
        key: str,
        max_requests: int,
        window_seconds: int
    ) -> bool:
        current = await self.redis.get(key)
        
        if not current:
            await self.redis.setex(key, window_seconds, 1)
            return False
        
        if int(current) >= max_requests:
            return True
        
        await self.redis.incr(key)
        return False

# RBAC (Role-Based Access Control)
class RBACPolicy:
    """Role-based access control implementation."""
    
    def __init__(self):
        self.role_permissions: Dict[str, List[str]] = {
            "admin": ["read", "write", "delete"],
            "user": ["read", "write"],
            "guest": ["read"]
        }
    
    def has_permission(
        self,
        roles: List[str],
        required_permission: str
    ) -> bool:
        for role in roles:
            if role in self.role_permissions:
                if required_permission in self.role_permissions[role]:
                    return True
        return False

# Security Service
class SecurityService:
    """Main security service implementation."""
    
    def __init__(
        self,
        password_hasher: PasswordHasher,
        token_provider: TokenProvider,
        rate_limiter: RateLimiter,
        rbac_policy: RBACPolicy
    ):
        self.password_hasher = password_hasher
        self.token_provider = token_provider
        self.rate_limiter = rate_limiter
        self.rbac_policy = rbac_policy
        self.users: Dict[str, User] = {}  # In-memory user store for demo
    
    async def register_user(
        self,
        username: str,
        email: str,
        password: str,
        roles: List[str] = ["user"]
    ) -> User:
        # Check if rate limited
        if await self.rate_limiter.is_rate_limited(
            f"register:{username}",
            max_requests=3,
            window_seconds=3600
        ):
            raise ValueError("Too many registration attempts")
        
        # Create user
        user = User(
            id=secrets.token_urlsafe(16),
            username=username,
            email=email,
            hashed_password=self.password_hasher.hash_password(password),
            roles=roles,
            is_active=True,
            created_at=datetime.utcnow()
        )
        
        self.users[user.id] = user
        return user
    
    async def authenticate(
        self,
        username: str,
        password: str
    ) -> Optional[Token]:
        # Check if rate limited
        if await self.rate_limiter.is_rate_limited(
            f"auth:{username}",
            max_requests=5,
            window_seconds=300
        ):
            raise ValueError("Too many login attempts")
        
        # Find user
        user = next(
            (u for u in self.users.values() if u.username == username),
            None
        )
        
        if not user or not user.is_active:
            return None
        
        # Verify password
        if not self.password_hasher.verify_password(
            password,
            user.hashed_password
        ):
            return None
        
        # Create tokens
        access_token = self.token_provider.create_access_token(
            data={"sub": user.id, "roles": user.roles},
            expires_delta=timedelta(minutes=15)
        )
        
        refresh_token = self.token_provider.create_refresh_token(
            data={"sub": user.id}
        )
        
        return Token(
            access_token=access_token,
            refresh_token=refresh_token
        )
    
    def verify_access(
        self,
        token: str,
        required_permission: str
    ) -> bool:
        try:
            # Verify token
            payload = self.token_provider.verify_token(token)
            
            # Check if token is refresh token
            if payload.get("refresh"):
                return False
            
            # Get user
            user = self.users.get(payload["sub"])
            if not user or not user.is_active:
                return False
            
            # Check permissions
            return self.rbac_policy.has_permission(
                user.roles,
                required_permission
            )
        
        except ValueError:
            return False

# FastAPI Implementation
app = FastAPI(title="Security Patterns Demo")

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Initialize services
password_hasher = BcryptPasswordHasher()
token_provider = JWTTokenProvider(secret_key="your-secret-key")
rate_limiter = RateLimiter(redis.Redis(host='localhost', port=6379, db=0))
rbac_policy = RBACPolicy()

security_service = SecurityService(
    password_hasher,
    token_provider,
    rate_limiter,
    rbac_policy
)

# API Models
class UserCreate(BaseModel):
    username: str
    email: EmailStr
    password: str

class UserResponse(BaseModel):
    id: str
    username: str
    email: str
    roles: List[str]
    created_at: datetime

# Dependencies
async def get_current_user(
    token: str = Depends(oauth2_scheme)
) -> User:
    try:
        payload = token_provider.verify_token(token)
        user = security_service.users.get(payload["sub"])
        if not user:
            raise HTTPException(status_code=401)
        return user
    except ValueError:
        raise HTTPException(status_code=401)

# API Routes
@app.post("/register", response_model=UserResponse)
async def register(user_data: UserCreate):
    try:
        user = await security_service.register_user(
            username=user_data.username,
            email=user_data.email,
            password=user_data.password
        )
        return user
    except ValueError as e:
        raise HTTPException(status_code=429, detail=str(e))

@app.post("/token")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    try:
        token = await security_service.authenticate(
            form_data.username,
            form_data.password
        )
        if not token:
            raise HTTPException(status_code=401)
        return token
    except ValueError as e:
        raise HTTPException(status_code=429, detail=str(e))

@app.get("/protected")
async def protected_route(
    user: User = Depends(get_current_user),
    permission: str = "read"
):
    if not security_service.rbac_policy.has_permission(
        user.roles,
        permission
    ):
        raise HTTPException(status_code=403)
    return {"message": "Access granted"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 