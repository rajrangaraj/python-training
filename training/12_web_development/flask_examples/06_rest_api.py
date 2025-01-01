"""
Demonstration of RESTful API development with Flask.
"""

from flask import Flask, request, jsonify, make_response
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import jwt
from functools import wraps
import uuid

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///api.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# Models
class User(db.Model):
    """User model for API authentication."""
    id = db.Column(db.Integer, primary_key=True)
    public_id = db.Column(db.String(50), unique=True)
    username = db.Column(db.String(50), unique=True)
    password_hash = db.Column(db.String(100))
    is_admin = db.Column(db.Boolean, default=False)

class Task(db.Model):
    """Task model for API resources."""
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(100))
    description = db.Column(db.String(200))
    done = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))

# Authentication decorator
def token_required(f):
    """Decorator to require valid JWT token."""
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('X-API-Token')
        
        if not token:
            return jsonify({'message': 'Token is missing'}), 401
        
        try:
            data = jwt.decode(token, app.config['SECRET_KEY'], algorithms=["HS256"])
            current_user = User.query.filter_by(public_id=data['public_id']).first()
        except:
            return jsonify({'message': 'Token is invalid'}), 401
        
        return f(current_user, *args, **kwargs)
    
    return decorated

# API Routes
@app.route('/api/auth/register', methods=['POST'])
def register():
    """Register a new user."""
    data = request.get_json()
    
    if not data or not data.get('username') or not data.get('password'):
        return jsonify({'message': 'Missing required fields'}), 400
    
    if User.query.filter_by(username=data['username']).first():
        return jsonify({'message': 'Username already exists'}), 400
    
    user = User(
        public_id=str(uuid.uuid4()),
        username=data['username'],
        password_hash=generate_password_hash(data['password'])
    )
    
    db.session.add(user)
    db.session.commit()
    
    return jsonify({'message': 'User created successfully'}), 201

@app.route('/api/auth/login', methods=['POST'])
def login():
    """Authenticate user and return token."""
    auth = request.authorization
    
    if not auth or not auth.username or not auth.password:
        return make_response(
            'Could not verify',
            401,
            {'WWW-Authenticate': 'Basic realm="Login required"'}
        )
    
    user = User.query.filter_by(username=auth.username).first()
    
    if not user or not check_password_hash(user.password_hash, auth.password):
        return make_response(
            'Could not verify',
            401,
            {'WWW-Authenticate': 'Basic realm="Login required"'}
        )
    
    token = jwt.encode(
        {
            'public_id': user.public_id,
            'exp': datetime.utcnow() + timedelta(hours=24)
        },
        app.config['SECRET_KEY'],
        algorithm="HS256"
    )
    
    return jsonify({'token': token})

# Task API endpoints
@app.route('/api/tasks', methods=['GET'])
@token_required
def get_tasks(current_user):
    """Get all tasks for current user."""
    tasks = Task.query.filter_by(user_id=current_user.id).all()
    
    return jsonify({
        'tasks': [{
            'id': task.id,
            'title': task.title,
            'description': task.description,
            'done': task.done,
            'created_at': task.created_at.isoformat()
        } for task in tasks]
    })

@app.route('/api/tasks', methods=['POST'])
@token_required
def create_task(current_user):
    """Create a new task."""
    data = request.get_json()
    
    if not data or not data.get('title'):
        return jsonify({'message': 'Missing required fields'}), 400
    
    task = Task(
        title=data['title'],
        description=data.get('description', ''),
        user_id=current_user.id
    )
    
    db.session.add(task)
    db.session.commit()
    
    return jsonify({
        'message': 'Task created',
        'task': {
            'id': task.id,
            'title': task.title,
            'description': task.description,
            'done': task.done,
            'created_at': task.created_at.isoformat()
        }
    }), 201

@app.route('/api/tasks/<int:task_id>', methods=['PUT'])
@token_required
def update_task(current_user, task_id):
    """Update a task."""
    task = Task.query.filter_by(id=task_id, user_id=current_user.id).first()
    
    if not task:
        return jsonify({'message': 'Task not found'}), 404
    
    data = request.get_json()
    
    if 'title' in data:
        task.title = data['title']
    if 'description' in data:
        task.description = data['description']
    if 'done' in data:
        task.done = bool(data['done'])
    
    db.session.commit()
    
    return jsonify({
        'message': 'Task updated',
        'task': {
            'id': task.id,
            'title': task.title,
            'description': task.description,
            'done': task.done,
            'created_at': task.created_at.isoformat()
        }
    })

@app.route('/api/tasks/<int:task_id>', methods=['DELETE'])
@token_required
def delete_task(current_user, task_id):
    """Delete a task."""
    task = Task.query.filter_by(id=task_id, user_id=current_user.id).first()
    
    if not task:
        return jsonify({'message': 'Task not found'}), 404
    
    db.session.delete(task)
    db.session.commit()
    
    return jsonify({'message': 'Task deleted'})

# Error handlers
@app.errorhandler(400)
def bad_request(error):
    """Handle bad request errors."""
    return jsonify({
        'error': 'Bad Request',
        'message': str(error)
    }), 400

@app.errorhandler(404)
def not_found(error):
    """Handle not found errors."""
    return jsonify({
        'error': 'Not Found',
        'message': str(error)
    }), 404

@app.errorhandler(500)
def server_error(error):
    """Handle internal server errors."""
    return jsonify({
        'error': 'Internal Server Error',
        'message': str(error)
    }), 500

if __name__ == '__main__':
    app.run(debug=True) 