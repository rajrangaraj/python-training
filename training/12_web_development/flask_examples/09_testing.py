"""
Demonstration of Flask application testing.
"""

from flask import Flask, jsonify, request
from flask_sqlalchemy import SQLAlchemy
import unittest
import json
import os

# Test application
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['TESTING'] = True
db = SQLAlchemy(app)

# Sample model
class User(db.Model):
    """User model for testing."""
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)

# Sample routes
@app.route('/api/users', methods=['GET'])
def get_users():
    """Get all users."""
    users = User.query.all()
    return jsonify([{
        'id': user.id,
        'username': user.username,
        'email': user.email
    } for user in users])

@app.route('/api/users', methods=['POST'])
def create_user():
    """Create a new user."""
    data = request.get_json()
    
    if not data or not all(k in data for k in ('username', 'email')):
        return jsonify({'error': 'Missing required fields'}), 400
    
    user = User(username=data['username'], email=data['email'])
    db.session.add(user)
    db.session.commit()
    
    return jsonify({
        'id': user.id,
        'username': user.username,
        'email': user.email
    }), 201

# Test cases
class TestFlaskAPI(unittest.TestCase):
    """Test cases for Flask API."""
    
    def setUp(self):
        """Set up test environment."""
        self.app = app.test_client()
        self.db = db
        
        with app.app_context():
            db.create_all()
    
    def tearDown(self):
        """Clean up test environment."""
        with app.app_context():
            db.session.remove()
            db.drop_all()
    
    def test_get_users_empty(self):
        """Test getting users when database is empty."""
        response = self.app.get('/api/users')
        data = json.loads(response.data)
        
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(data), 0)
    
    def test_create_user(self):
        """Test user creation."""
        user_data = {
            'username': 'testuser',
            'email': 'test@example.com'
        }
        
        response = self.app.post(
            '/api/users',
            data=json.dumps(user_data),
            content_type='application/json'
        )
        data = json.loads(response.data)
        
        self.assertEqual(response.status_code, 201)
        self.assertEqual(data['username'], user_data['username'])
        self.assertEqual(data['email'], user_data['email'])
    
    def test_create_user_missing_fields(self):
        """Test user creation with missing fields."""
        user_data = {'username': 'testuser'}
        
        response = self.app.post(
            '/api/users',
            data=json.dumps(user_data),
            content_type='application/json'
        )
        data = json.loads(response.data)
        
        self.assertEqual(response.status_code, 400)
        self.assertEqual(data['error'], 'Missing required fields')
    
    def test_get_users_after_create(self):
        """Test getting users after creating one."""
        # Create a user first
        user_data = {
            'username': 'testuser',
            'email': 'test@example.com'
        }
        self.app.post(
            '/api/users',
            data=json.dumps(user_data),
            content_type='application/json'
        )
        
        # Get all users
        response = self.app.get('/api/users')
        data = json.loads(response.data)
        
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]['username'], user_data['username'])

# Integration tests
class TestFlaskIntegration(unittest.TestCase):
    """Integration tests for Flask application."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test database."""
        app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///test.db'
        cls.app = app.test_client()
        
        with app.app_context():
            db.create_all()
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test database."""
        with app.app_context():
            db.session.remove()
            db.drop_all()
        
        # Remove test database file
        os.remove('test.db')
    
    def setUp(self):
        """Set up test case."""
        with app.app_context():
            db.session.query(User).delete()
            db.session.commit()
    
    def test_user_workflow(self):
        """Test complete user workflow."""
        # Create first user
        user1_data = {
            'username': 'user1',
            'email': 'user1@example.com'
        }
        response1 = self.app.post(
            '/api/users',
            data=json.dumps(user1_data),
            content_type='application/json'
        )
        self.assertEqual(response1.status_code, 201)
        
        # Create second user
        user2_data = {
            'username': 'user2',
            'email': 'user2@example.com'
        }
        response2 = self.app.post(
            '/api/users',
            data=json.dumps(user2_data),
            content_type='application/json'
        )
        self.assertEqual(response2.status_code, 201)
        
        # Get all users
        response3 = self.app.get('/api/users')
        data = json.loads(response3.data)
        
        self.assertEqual(response3.status_code, 200)
        self.assertEqual(len(data), 2)
        usernames = [user['username'] for user in data]
        self.assertIn('user1', usernames)
        self.assertIn('user2', usernames)

if __name__ == '__main__':
    unittest.main() 