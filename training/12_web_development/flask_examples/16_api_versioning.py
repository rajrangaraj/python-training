"""
Demonstration of Flask API versioning and documentation.
"""

from flask import Flask, jsonify, request, url_for
from flask_restx import Api, Resource, fields, Namespace
from functools import wraps
import re
from datetime import datetime
from enum import Enum
import marshmallow as ma
from dataclasses import dataclass
from typing import List, Optional

app = Flask(__name__)

# API versioning configuration
class ApiVersion(Enum):
    V1 = '1'
    V2 = '2'
    V3 = '3'

# Create versioned APIs
api_v1 = Api(
    app,
    version='1.0',
    title='My API',
    description='API Version 1',
    doc='/docs/v1',
    prefix='/api/v1'
)

api_v2 = Api(
    app,
    version='2.0',
    title='My API',
    description='API Version 2 with enhanced features',
    doc='/docs/v2',
    prefix='/api/v2'
)

# Namespaces
users_v1 = Namespace('users', description='User operations V1')
users_v2 = Namespace('users', description='User operations V2')
posts_v1 = Namespace('posts', description='Post operations V1')
posts_v2 = Namespace('posts', description='Post operations V2')

api_v1.add_namespace(users_v1)
api_v1.add_namespace(posts_v1)
api_v2.add_namespace(users_v2)
api_v2.add_namespace(posts_v2)

# Data models
@dataclass
class User:
    id: int
    username: str
    email: str
    created_at: datetime
    posts: List['Post']

@dataclass
class Post:
    id: int
    title: str
    content: str
    author_id: int
    created_at: datetime
    updated_at: Optional[datetime]

# API Models
user_model_v1 = api_v1.model('User', {
    'id': fields.Integer(readonly=True),
    'username': fields.String(required=True),
    'email': fields.String(required=True)
})

user_model_v2 = api_v2.model('User', {
    'id': fields.Integer(readonly=True),
    'username': fields.String(required=True),
    'email': fields.String(required=True),
    'created_at': fields.DateTime(readonly=True),
    'posts_count': fields.Integer(readonly=True)
})

post_model_v1 = api_v1.model('Post', {
    'id': fields.Integer(readonly=True),
    'title': fields.String(required=True),
    'content': fields.String(required=True)
})

post_model_v2 = api_v2.model('Post', {
    'id': fields.Integer(readonly=True),
    'title': fields.String(required=True),
    'content': fields.String(required=True),
    'author': fields.Nested(user_model_v2),
    'created_at': fields.DateTime(readonly=True),
    'updated_at': fields.DateTime(readonly=True)
})

# Schemas for validation
class UserSchemaV1(ma.Schema):
    class Meta:
        fields = ('id', 'username', 'email')
    
    username = ma.fields.String(required=True, validate=[
        ma.validate.Length(min=3, max=50),
        ma.validate.Regexp('^[a-zA-Z0-9_]+$')
    ])
    email = ma.fields.Email(required=True)

class UserSchemaV2(UserSchemaV1):
    class Meta:
        fields = ('id', 'username', 'email', 'created_at', 'posts_count')
    
    created_at = ma.fields.DateTime(dump_only=True)
    posts_count = ma.fields.Integer(dump_only=True)

# API Resources
@users_v1.route('/')
class UsersV1(Resource):
    @users_v1.doc('list_users')
    @users_v1.marshal_list_with(user_model_v1)
    def get(self):
        """List all users (V1)"""
        return list(users.values())
    
    @users_v1.doc('create_user')
    @users_v1.expect(user_model_v1)
    @users_v1.marshal_with(user_model_v1, code=201)
    def post(self):
        """Create a new user (V1)"""
        schema = UserSchemaV1()
        data = schema.load(request.json)
        user = User(
            id=len(users) + 1,
            username=data['username'],
            email=data['email'],
            created_at=datetime.utcnow(),
            posts=[]
        )
        users[user.id] = user
        return user, 201

@users_v2.route('/')
class UsersV2(Resource):
    @users_v2.doc('list_users')
    @users_v2.marshal_list_with(user_model_v2)
    def get(self):
        """List all users with enhanced details (V2)"""
        return [
            {
                **user.__dict__,
                'posts_count': len(user.posts)
            }
            for user in users.values()
        ]
    
    @users_v2.doc('create_user')
    @users_v2.expect(user_model_v2)
    @users_v2.marshal_with(user_model_v2, code=201)
    def post(self):
        """Create a new user with enhanced validation (V2)"""
        schema = UserSchemaV2()
        data = schema.load(request.json)
        user = User(
            id=len(users) + 1,
            username=data['username'],
            email=data['email'],
            created_at=datetime.utcnow(),
            posts=[]
        )
        users[user.id] = user
        return {
            **user.__dict__,
            'posts_count': 0
        }, 201

# API documentation
@app.route('/api')
def api_documentation():
    """API documentation hub."""
    return jsonify({
        'versions': {
            'v1': url_for('api_v1.doc'),
            'v2': url_for('api_v2.doc')
        },
        'latest': url_for('api_v2.doc'),
        'deprecated': ['v1'],
        'sunset_dates': {
            'v1': '2024-12-31'
        }
    })

# Version deprecation decorator
def deprecated_version(version: ApiVersion, sunset_date: str):
    def decorator(f):
        @wraps(f)
        def wrapped(*args, **kwargs):
            response = f(*args, **kwargs)
            if isinstance(response, tuple):
                response, status_code = response
            else:
                status_code = 200
            
            headers = {
                'Deprecation': f'version="{version.value}"',
                'Sunset': sunset_date,
                'Link': f'</api/v{int(version.value) + 1}>; rel="successor-version"'
            }
            
            if isinstance(response, dict):
                response = jsonify(response)
            
            for key, value in headers.items():
                response.headers[key] = value
            
            return response, status_code
        return wrapped
    return decorator

# Example of deprecated endpoint
@users_v1.route('/<int:id>')
class UserV1(Resource):
    @users_v1.doc('get_user')
    @users_v1.marshal_with(user_model_v1)
    @deprecated_version(ApiVersion.V1, '2024-12-31')
    def get(self, id):
        """Get user by ID (V1 - Deprecated)"""
        return users.get(id)

if __name__ == '__main__':
    app.run(debug=True) 