"""
Demonstration of Flask WebSocket integration and real-time features.
"""

from flask import Flask, render_template
from flask_socketio import SocketIO, emit, join_room, leave_room
from dataclasses import dataclass
from typing import Dict, Set, Optional
from datetime import datetime
import json
import asyncio
import functools
import logging

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, cors_allowed_origins="*")

# Data models
@dataclass
class ChatRoom:
    id: str
    name: str
    created_at: datetime
    users: Set[str]
    messages: list

@dataclass
class ChatMessage:
    id: int
    room_id: str
    user_id: str
    content: str
    timestamp: datetime

# In-memory storage
chat_rooms: Dict[str, ChatRoom] = {}
user_sessions: Dict[str, str] = {}  # user_id -> session_id
message_counter = 0

# Event handlers
@socketio.on('connect')
def handle_connect():
    """Handle client connection."""
    logging.info(f'Client connected: {request.sid}')
    emit('connection_established', {'status': 'connected'})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection."""
    user_id = None
    for uid, sid in user_sessions.items():
        if sid == request.sid:
            user_id = uid
            break
    
    if user_id:
        for room in chat_rooms.values():
            if user_id in room.users:
                room.users.remove(user_id)
                emit('user_left', {
                    'user_id': user_id,
                    'room_id': room.id
                }, room=room.id)
        
        del user_sessions[user_id]
    
    logging.info(f'Client disconnected: {request.sid}')

@socketio.on('join')
def handle_join(data):
    """Handle room join request."""
    room_id = data.get('room_id')
    user_id = data.get('user_id')
    
    if not room_id or not user_id:
        emit('error', {'message': 'Invalid join request'})
        return
    
    room = chat_rooms.get(room_id)
    if not room:
        emit('error', {'message': 'Room not found'})
        return
    
    join_room(room_id)
    room.users.add(user_id)
    user_sessions[user_id] = request.sid
    
    emit('joined', {
        'room_id': room_id,
        'user_id': user_id,
        'active_users': list(room.users)
    }, room=room_id)
    
    # Send recent messages
    emit('message_history', {
        'messages': [
            {
                'id': msg.id,
                'user_id': msg.user_id,
                'content': msg.content,
                'timestamp': msg.timestamp.isoformat()
            }
            for msg in room.messages[-50:]  # Last 50 messages
        ]
    })

@socketio.on('leave')
def handle_leave(data):
    """Handle room leave request."""
    room_id = data.get('room_id')
    user_id = data.get('user_id')
    
    if not room_id or not user_id:
        return
    
    room = chat_rooms.get(room_id)
    if room and user_id in room.users:
        leave_room(room_id)
        room.users.remove(user_id)
        emit('user_left', {
            'user_id': user_id,
            'room_id': room_id
        }, room=room_id)

@socketio.on('message')
def handle_message(data):
    """Handle new message."""
    global message_counter
    room_id = data.get('room_id')
    user_id = data.get('user_id')
    content = data.get('content')
    
    if not all([room_id, user_id, content]):
        emit('error', {'message': 'Invalid message format'})
        return
    
    room = chat_rooms.get(room_id)
    if not room or user_id not in room.users:
        emit('error', {'message': 'Not in room'})
        return
    
    message_counter += 1
    message = ChatMessage(
        id=message_counter,
        room_id=room_id,
        user_id=user_id,
        content=content,
        timestamp=datetime.utcnow()
    )
    
    room.messages.append(message)
    
    emit('new_message', {
        'id': message.id,
        'user_id': message.user_id,
        'content': message.content,
        'timestamp': message.timestamp.isoformat()
    }, room=room_id)

@socketio.on('typing')
def handle_typing(data):
    """Handle typing indicator."""
    room_id = data.get('room_id')
    user_id = data.get('user_id')
    is_typing = data.get('typing', False)
    
    if room_id and user_id:
        emit('user_typing', {
            'user_id': user_id,
            'typing': is_typing
        }, room=room_id, include_self=False)

# Room management
@app.route('/api/rooms', methods=['POST'])
def create_room():
    """Create a new chat room."""
    data = request.json
    room_id = data.get('id')
    room_name = data.get('name')
    
    if not room_id or not room_name:
        return jsonify({'error': 'Invalid room data'}), 400
    
    if room_id in chat_rooms:
        return jsonify({'error': 'Room already exists'}), 409
    
    room = ChatRoom(
        id=room_id,
        name=room_name,
        created_at=datetime.utcnow(),
        users=set(),
        messages=[]
    )
    chat_rooms[room_id] = room
    
    return jsonify({
        'id': room.id,
        'name': room.name,
        'created_at': room.created_at.isoformat()
    }), 201

@app.route('/api/rooms/<room_id>')
def get_room(room_id):
    """Get room details."""
    room = chat_rooms.get(room_id)
    if not room:
        return jsonify({'error': 'Room not found'}), 404
    
    return jsonify({
        'id': room.id,
        'name': room.name,
        'created_at': room.created_at.isoformat(),
        'active_users': len(room.users),
        'message_count': len(room.messages)
    })

# Error handling
@socketio.on_error()
def error_handler(e):
    """Handle WebSocket errors."""
    logging.error(f'WebSocket error: {str(e)}')
    emit('error', {'message': 'Internal server error'})

# Client example
JAVASCRIPT_CLIENT = """
const socket = io('http://localhost:5000');

socket.on('connect', () => {
    console.log('Connected to server');
});

socket.on('new_message', (message) => {
    console.log('New message:', message);
});

socket.on('user_typing', (data) => {
    console.log(`User ${data.user_id} is ${data.typing ? 'typing' : 'not typing'}`);
});

function joinRoom(roomId, userId) {
    socket.emit('join', { room_id: roomId, user_id: userId });
}

function sendMessage(roomId, userId, content) {
    socket.emit('message', {
        room_id: roomId,
        user_id: userId,
        content: content
    });
}

function indicateTyping(roomId, userId, isTyping) {
    socket.emit('typing', {
        room_id: roomId,
        user_id: userId,
        typing: isTyping
    });
}
"""

if __name__ == '__main__':
    socketio.run(app, debug=True) 