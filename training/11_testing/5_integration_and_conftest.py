"""
Demonstration of integration testing and test configuration using conftest.
"""

import pytest
import sqlite3
from typing import List, Dict, Any, Generator
from dataclasses import dataclass
from datetime import datetime
import json
import os

# Domain models
@dataclass
class User:
    """User model."""
    id: int
    username: str
    email: str
    created_at: datetime

@dataclass
class Post:
    """Blog post model."""
    id: int
    user_id: int
    title: str
    content: str
    created_at: datetime

# Database layer
class Database:
    """Database interface for the application."""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self._create_tables()
    
    def _create_tables(self) -> None:
        """Create necessary database tables."""
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            CREATE TABLE IF NOT EXISTS posts (
                id INTEGER PRIMARY KEY,
                user_id INTEGER NOT NULL,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            );
        """)
        self.conn.commit()
    
    def close(self) -> None:
        """Close database connection."""
        self.conn.close()

# Repository layer
class UserRepository:
    """Repository for user operations."""
    
    def __init__(self, db: Database):
        self.db = db
    
    def create_user(self, username: str, email: str) -> User:
        """Create a new user."""
        cursor = self.db.conn.execute(
            "INSERT INTO users (username, email) VALUES (?, ?)",
            (username, email)
        )
        self.db.conn.commit()
        
        return self.get_user_by_id(cursor.lastrowid)
    
    def get_user_by_id(self, user_id: int) -> User:
        """Get user by ID."""
        row = self.db.conn.execute(
            "SELECT * FROM users WHERE id = ?",
            (user_id,)
        ).fetchone()
        
        if not row:
            raise ValueError(f"User {user_id} not found")
        
        return User(
            id=row['id'],
            username=row['username'],
            email=row['email'],
            created_at=datetime.fromisoformat(row['created_at'])
        )

class PostRepository:
    """Repository for post operations."""
    
    def __init__(self, db: Database):
        self.db = db
    
    def create_post(self, user_id: int, title: str, content: str) -> Post:
        """Create a new post."""
        cursor = self.db.conn.execute(
            "INSERT INTO posts (user_id, title, content) VALUES (?, ?, ?)",
            (user_id, title, content)
        )
        self.db.conn.commit()
        
        return self.get_post_by_id(cursor.lastrowid)
    
    def get_post_by_id(self, post_id: int) -> Post:
        """Get post by ID."""
        row = self.db.conn.execute(
            "SELECT * FROM posts WHERE id = ?",
            (post_id,)
        ).fetchone()
        
        if not row:
            raise ValueError(f"Post {post_id} not found")
        
        return Post(
            id=row['id'],
            user_id=row['user_id'],
            title=row['title'],
            content=row['content'],
            created_at=datetime.fromisoformat(row['created_at'])
        )
    
    def get_user_posts(self, user_id: int) -> List[Post]:
        """Get all posts by a user."""
        rows = self.db.conn.execute(
            "SELECT * FROM posts WHERE user_id = ? ORDER BY created_at DESC",
            (user_id,)
        ).fetchall()
        
        return [
            Post(
                id=row['id'],
                user_id=row['user_id'],
                title=row['title'],
                content=row['content'],
                created_at=datetime.fromisoformat(row['created_at'])
            )
            for row in rows
        ]

# Service layer
class BlogService:
    """Service layer for blog operations."""
    
    def __init__(self, user_repo: UserRepository, post_repo: PostRepository):
        self.user_repo = user_repo
        self.post_repo = post_repo
    
    def create_user_with_post(
        self,
        username: str,
        email: str,
        post_title: str,
        post_content: str
    ) -> Dict[str, Any]:
        """Create a new user and their first post."""
        user = self.user_repo.create_user(username, email)
        post = self.post_repo.create_post(user.id, post_title, post_content)
        
        return {
            "user": user,
            "post": post
        }
    
    def get_user_blog_data(self, user_id: int) -> Dict[str, Any]:
        """Get user and all their posts."""
        user = self.user_repo.get_user_by_id(user_id)
        posts = self.post_repo.get_user_posts(user_id)
        
        return {
            "user": user,
            "posts": posts
        }

# Integration tests
class TestBlogIntegration:
    """Integration tests for blog functionality."""
    
    def test_create_user_with_post(self, blog_service: BlogService):
        """Test creating a user with their first post."""
        result = blog_service.create_user_with_post(
            "testuser",
            "test@example.com",
            "First Post",
            "Hello, World!"
        )
        
        assert result["user"].username == "testuser"
        assert result["post"].title == "First Post"
        
        # Verify data persistence
        blog_data = blog_service.get_user_blog_data(result["user"].id)
        assert len(blog_data["posts"]) == 1
        assert blog_data["posts"][0].content == "Hello, World!"
    
    def test_multiple_posts(self, blog_service: BlogService):
        """Test creating multiple posts for a user."""
        # Create user with first post
        result = blog_service.create_user_with_post(
            "multipost",
            "multi@example.com",
            "Post 1",
            "First content"
        )
        user_id = result["user"].id
        
        # Add more posts
        post2 = blog_service.post_repo.create_post(
            user_id,
            "Post 2",
            "Second content"
        )
        post3 = blog_service.post_repo.create_post(
            user_id,
            "Post 3",
            "Third content"
        )
        
        # Verify all posts
        blog_data = blog_service.get_user_blog_data(user_id)
        assert len(blog_data["posts"]) == 3
        assert [p.title for p in blog_data["posts"]] == [
            "Post 3",
            "Post 2",
            "Post 1"
        ]
    
    def test_user_not_found(self, blog_service: BlogService):
        """Test handling of non-existent user."""
        with pytest.raises(ValueError, match="User 999 not found"):
            blog_service.get_user_blog_data(999)

if __name__ == '__main__':
    pytest.main([__file__]) 