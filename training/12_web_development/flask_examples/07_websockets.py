"""
Demonstration of WebSocket integration with Flask using Flask-SocketIO.
"""

from flask import Flask, render_template
from flask_socketio import SocketIO, emit, join_room, leave_room
from datetime import datetime
import json

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'
socketio = SocketIO(app)

# In-memory storage for demo purposes
active_users = {}
chat_rooms = {
    'general': [],
    'tech': [],
    'random': []
}

class ChatMessage:
    """Chat message container."""
    def __init__(self, username, message, room, timestamp=None):
        self.username = username
        self.message = message
        self.room = room
        self.timestamp = timestamp or datetime.utcnow()
    
    def to_dict(self):
        """Convert message to dictionary."""
        return {
            'username': self.username,
            'message': self.message,
            'room': self.room,
            'timestamp': self.timestamp.isoformat()
        }

@app.route('/')
def index():
    """Render chat application page."""
    return render_template('chat/index.html', rooms=chat_rooms.keys())

@socketio.on('connect')
def handle_connect():
    """Handle client connection."""
    emit('status', {'message': 'Connected to server'})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection."""
    if 'username' in active_users:
        username = active_users.pop('username')
        for room in chat_rooms:
            if username in chat_rooms[room]:
                chat_rooms[room].remove(username)
                emit('user_left', {'username': username}, room=room)

@socketio.on('join')
def handle_join(data):
    """Handle user joining a chat room."""
    username = data.get('username')
    room = data.get('room')
    
    if not username or not room:
        return
    
    active_users[username] = request.sid
    join_room(room)
    
    if username not in chat_rooms[room]:
        chat_rooms[room].append(username)
    
    message = ChatMessage(
        username='System',
        message=f'{username} has joined the room',
        room=room
    )
    
    emit('user_joined', {
        'username': username,
        'active_users': chat_rooms[room]
    }, room=room)
    
    emit('message', message.to_dict(), room=room)

@socketio.on('leave')
def handle_leave(data):
    """Handle user leaving a chat room."""
    username = data.get('username')
    room = data.get('room')
    
    if not username or not room:
        return
    
    leave_room(room)
    if username in chat_rooms[room]:
        chat_rooms[room].remove(username)
    
    message = ChatMessage(
        username='System',
        message=f'{username} has left the room',
        room=room
    )
    
    emit('user_left', {
        'username': username,
        'active_users': chat_rooms[room]
    }, room=room)
    
    emit('message', message.to_dict(), room=room)

@socketio.on('message')
def handle_message(data):
    """Handle chat messages."""
    username = data.get('username')
    message_text = data.get('message')
    room = data.get('room')
    
    if not all([username, message_text, room]):
        return
    
    message = ChatMessage(username, message_text, room)
    emit('message', message.to_dict(), room=room)

@socketio.on('typing')
def handle_typing(data):
    """Handle typing notifications."""
    username = data.get('username')
    room = data.get('room')
    is_typing = data.get('typing', False)
    
    if not username or not room:
        return
    
    emit('typing', {
        'username': username,
        'typing': is_typing
    }, room=room)

if __name__ == '__main__':
    socketio.run(app, debug=True) 