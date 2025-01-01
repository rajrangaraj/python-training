"""
Test configuration and fixtures for the blog application tests.
"""

import pytest
import os
from typing import Generator
from .integration_and_conftest import Database, UserRepository, PostRepository, BlogService

@pytest.fixture(scope="session")
def test_db() -> Generator[Database, None, None]:
    """Create a test database for the test session."""
    db_path = "test_blog.db"
    
    # Ensure clean database
    if os.path.exists(db_path):
        os.remove(db_path)
    
    db = Database(db_path)
    yield db
    
    # Cleanup
    db.close()
    os.remove(db_path)

@pytest.fixture
def user_repository(test_db: Database) -> UserRepository:
    """Provide a UserRepository instance."""
    return UserRepository(test_db)

@pytest.fixture
def post_repository(test_db: Database) -> PostRepository:
    """Provide a PostRepository instance."""
    return PostRepository(test_db)

@pytest.fixture
def blog_service(
    user_repository: UserRepository,
    post_repository: PostRepository
) -> BlogService:
    """Provide a BlogService instance."""
    return BlogService(user_repository, post_repository) 