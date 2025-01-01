"""
Demonstration of testing asynchronous code and API endpoints using pytest-asyncio and pytest-aiohttp.
"""

import pytest
import aiohttp
from aiohttp import web
from typing import Dict, List, Optional, Callable
import asyncio
from dataclasses import dataclass
import json
from datetime import datetime

# Models
@dataclass
class Task:
    """Task model for the API."""
    id: int
    title: str
    completed: bool
    created_at: datetime

# In-memory storage
class TaskStore:
    """Task storage implementation."""
    
    def __init__(self):
        self.tasks: Dict[int, Task] = {}
        self.next_id: int = 1
    
    async def add_task(self, title: str) -> Task:
        """Add a new task."""
        task = Task(
            id=self.next_id,
            title=title,
            completed=False,
            created_at=datetime.now()
        )
        self.tasks[task.id] = task
        self.next_id += 1
        return task
    
    async def get_task(self, task_id: int) -> Optional[Task]:
        """Get task by ID."""
        return self.tasks.get(task_id)
    
    async def list_tasks(self) -> List[Task]:
        """List all tasks."""
        return list(self.tasks.values())
    
    async def update_task(self, task_id: int, completed: bool) -> Optional[Task]:
        """Update task completion status."""
        if task := self.tasks.get(task_id):
            task.completed = completed
            return task
        return None

# API Routes
async def create_task(request: web.Request) -> web.Response:
    """Create a new task."""
    store: TaskStore = request.app['store']
    data = await request.json()
    
    if 'title' not in data:
        return web.json_response(
            {'error': 'title is required'},
            status=400
        )
    
    task = await store.add_task(data['title'])
    return web.json_response({
        'id': task.id,
        'title': task.title,
        'completed': task.completed,
        'created_at': task.created_at.isoformat()
    })

async def get_task(request: web.Request) -> web.Response:
    """Get task by ID."""
    store: TaskStore = request.app['store']
    task_id = int(request.match_info['id'])
    
    if task := await store.get_task(task_id):
        return web.json_response({
            'id': task.id,
            'title': task.title,
            'completed': task.completed,
            'created_at': task.created_at.isoformat()
        })
    
    return web.json_response(
        {'error': 'Task not found'},
        status=404
    )

# Application factory
def create_app(store: Optional[TaskStore] = None) -> web.Application:
    """Create the web application."""
    app = web.Application()
    app['store'] = store or TaskStore()
    
    app.router.add_post('/tasks', create_task)
    app.router.add_get('/tasks/{id}', get_task)
    
    return app

# Tests
class TestTaskStore:
    """Tests for the TaskStore class."""
    
    @pytest.fixture
    async def store(self) -> TaskStore:
        """Provide a fresh TaskStore instance."""
        return TaskStore()
    
    @pytest.mark.asyncio
    async def test_add_task(self, store: TaskStore):
        """Test adding a task."""
        task = await store.add_task("Test task")
        assert task.id == 1
        assert task.title == "Test task"
        assert not task.completed
    
    @pytest.mark.asyncio
    async def test_get_task(self, store: TaskStore):
        """Test getting a task."""
        task = await store.add_task("Test task")
        retrieved = await store.get_task(task.id)
        assert retrieved == task
    
    @pytest.mark.asyncio
    async def test_list_tasks(self, store: TaskStore):
        """Test listing tasks."""
        task1 = await store.add_task("Task 1")
        task2 = await store.add_task("Task 2")
        
        tasks = await store.list_tasks()
        assert len(tasks) == 2
        assert tasks == [task1, task2]

class TestTaskAPI:
    """Integration tests for the Task API."""
    
    @pytest.fixture
    async def cli(self, aiohttp_client) -> aiohttp.ClientSession:
        """Provide a test client."""
        app = create_app()
        return await aiohttp_client(app)
    
    @pytest.mark.asyncio
    async def test_create_task(self, cli):
        """Test task creation endpoint."""
        resp = await cli.post('/tasks', json={'title': 'New task'})
        assert resp.status == 200
        
        data = await resp.json()
        assert data['title'] == 'New task'
        assert not data['completed']
    
    @pytest.mark.asyncio
    async def test_create_task_invalid(self, cli):
        """Test task creation with invalid data."""
        resp = await cli.post('/tasks', json={})
        assert resp.status == 400
        
        data = await resp.json()
        assert 'error' in data
    
    @pytest.mark.asyncio
    async def test_get_task(self, cli):
        """Test getting a task."""
        # Create a task first
        create_resp = await cli.post('/tasks', json={'title': 'Test task'})
        create_data = await create_resp.json()
        task_id = create_data['id']
        
        # Get the task
        get_resp = await cli.get(f'/tasks/{task_id}')
        assert get_resp.status == 200
        
        get_data = await get_resp.json()
        assert get_data['id'] == task_id
        assert get_data['title'] == 'Test task'
    
    @pytest.mark.asyncio
    async def test_get_task_not_found(self, cli):
        """Test getting a non-existent task."""
        resp = await cli.get('/tasks/999')
        assert resp.status == 404

if __name__ == '__main__':
    pytest.main([__file__]) 