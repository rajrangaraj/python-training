"""
Demonstration of Flask testing with pytest.
"""

import pytest
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
import json

# Test fixtures
@pytest.fixture
def app():
    """Create test application."""
    app = Flask(__name__)
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    app.config['TESTING'] = True
    
    return app

@pytest.fixture
def db(app):
    """Create test database."""
    db = SQLAlchemy(app)
    
    # Define User model
    class User(db.Model):
        id = db.Column(db.Integer, primary_key=True)
        username = db.Column(db.String(80), unique=True, nullable=False)
        email = db.Column(db.String(120), unique=True, nullable=False)
    
    # Create tables
    with app.app_context():
        db.create_all()
    
    return db

@pytest.fixture
def client(app):
    """Create test client."""
    return app.test_client()

# Test cases
def test_get_users_empty(client):
    """Test getting users when database is empty."""
    response = client.get('/api/users')
    data = json.loads(response.data)
    
    assert response.status_code == 200
    assert len(data) == 0

def test_create_user(client):
    """Test user creation."""
    user_data = {
        'username': 'testuser',
        'email': 'test@example.com'
    }
    
    response = client.post(
        '/api/users',
        data=json.dumps(user_data),
        content_type='application/json'
    )
    data = json.loads(response.data)
    
    assert response.status_code == 201
    assert data['username'] == user_data['username']
    assert data['email'] == user_data['email']

def test_create_user_missing_fields(client):
    """Test user creation with missing fields."""
    user_data = {'username': 'testuser'}
    
    response = client.post(
        '/api/users',
        data=json.dumps(user_data),
        content_type='application/json'
    )
    data = json.loads(response.data)
    
    assert response.status_code == 400
    assert data['error'] == 'Missing required fields'

@pytest.mark.integration
def test_user_workflow(client, db):
    """Test complete user workflow."""
    # Create first user
    user1_data = {
        'username': 'user1',
        'email': 'user1@example.com'
    }
    response1 = client.post(
        '/api/users',
        data=json.dumps(user1_data),
        content_type='application/json'
    )
    assert response1.status_code == 201
    
    # Create second user
    user2_data = {
        'username': 'user2',
        'email': 'user2@example.com'
    }
    response2 = client.post(
        '/api/users',
        data=json.dumps(user2_data),
        content_type='application/json'
    )
    assert response2.status_code == 201
    
    # Get all users
    response3 = client.get('/api/users')
    data = json.loads(response3.data)
    
    assert response3.status_code == 200
    assert len(data) == 2
    usernames = [user['username'] for user in data]
    assert 'user1' in usernames
    assert 'user2' in usernames 