"""
Demonstration of pytest plugins and advanced fixture patterns.
"""

import pytest
from typing import Dict, List, Generator, Any
import tempfile
import os
import json
import logging
from datetime import datetime, timedelta
from contextlib import contextmanager

# Custom pytest plugin
def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers",
        "integration: mark test as integration test"
    )

# Fixtures with different scopes
@pytest.fixture(scope="session")
def test_data() -> Dict[str, Any]:
    """Provide test data for the entire test session."""
    return {
        "users": [
            {"id": 1, "name": "Alice", "role": "admin"},
            {"id": 2, "name": "Bob", "role": "user"},
            {"id": 3, "name": "Charlie", "role": "user"}
        ],
        "settings": {
            "timeout": 30,
            "retry_count": 3,
            "cache_ttl": 300
        }
    }

@pytest.fixture(scope="module")
def temp_dir() -> Generator[str, None, None]:
    """Provide a temporary directory for the test module."""
    with tempfile.TemporaryDirectory() as tmpdirname:
        yield tmpdirname

@pytest.fixture
def logger() -> Generator[logging.Logger, None, None]:
    """Provide a configured logger for each test."""
    logger = logging.getLogger("test_logger")
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    
    yield logger
    
    # Cleanup
    logger.handlers.clear()

# Test helper classes
class TestContext:
    """Context manager for test setup and cleanup."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.start_time = None
    
    def __enter__(self):
        self.start_time = datetime.now()
        self.logger.info(f"Test started at {self.start_time}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = datetime.now()
        duration = end_time - self.start_time
        self.logger.info(f"Test ended at {end_time} (duration: {duration})")
        
        if exc_type:
            self.logger.error(f"Test failed: {exc_val}")

@contextmanager
def data_file(temp_dir: str, filename: str, data: Dict) -> Generator[str, None, None]:
    """Create a temporary data file."""
    filepath = os.path.join(temp_dir, filename)
    with open(filepath, 'w') as f:
        json.dump(data, f)
    
    yield filepath
    
    os.remove(filepath)

# Test classes
class TestWithPlugins:
    """Test class demonstrating plugin usage."""
    
    @pytest.mark.slow
    def test_slow_operation(self, logger):
        """A slow test that demonstrates custom markers."""
        with TestContext(logger):
            # Simulate slow operation
            import time
            time.sleep(1)
            assert True

    @pytest.mark.integration
    def test_with_test_data(self, test_data, temp_dir, logger):
        """Test using session and module scoped fixtures."""
        with TestContext(logger):
            # Create temporary data file
            with data_file(temp_dir, "users.json", test_data["users"]) as filepath:
                # Read and verify data
                with open(filepath) as f:
                    data = json.load(f)
                assert len(data) == 3
                assert data[0]["name"] == "Alice"

    def test_fixture_isolation(self, temp_dir, logger):
        """Test demonstrating fixture isolation."""
        with TestContext(logger):
            test_file = os.path.join(temp_dir, "test.txt")
            with open(test_file, 'w') as f:
                f.write("test data")
            
            assert os.path.exists(test_file)
            with open(test_file) as f:
                assert f.read() == "test data"

# Parametrized tests with custom markers
@pytest.mark.integration
@pytest.mark.parametrize("user_id,expected_role", [
    (1, "admin"),
    (2, "user"),
    (3, "user")
])
def test_user_roles(test_data, user_id, expected_role):
    """Test user roles with parameterization."""
    user = next(u for u in test_data["users"] if u["id"] == user_id)
    assert user["role"] == expected_role

# Test using multiple fixtures
def test_complex_setup(test_data, temp_dir, logger):
    """Test demonstrating complex fixture interactions."""
    with TestContext(logger) as context:
        # Create multiple data files
        files = []
        try:
            # Create user data file
            with data_file(temp_dir, "users.json", test_data["users"]) as user_file:
                files.append(user_file)
                # Create settings file
                with data_file(temp_dir, "settings.json", test_data["settings"]) as settings_file:
                    files.append(settings_file)
                    
                    # Verify both files exist
                    for file in files:
                        assert os.path.exists(file)
                        with open(file) as f:
                            data = json.load(f)
                            assert isinstance(data, (list, dict))
        
        except Exception as e:
            logger.error(f"Test failed during file operations: {e}")
            raise

if __name__ == '__main__':
    pytest.main([__file__, "-v"]) 