"""
Basic Flask routing and request handling demonstration.
"""

from flask import Flask, request, jsonify

app = Flask(__name__)

# Basic routes
@app.route('/')
def home():
    """Simple home page."""
    return '<h1>Welcome to Flask!</h1>'

@app.route('/hello/<name>')
def hello(name):
    """Dynamic route with URL parameter."""
    return f'Hello, {name}!'

# HTTP methods
@app.route('/api/echo', methods=['GET', 'POST'])
def echo():
    """Echo back request data."""
    if request.method == 'POST':
        return jsonify({
            'method': 'POST',
            'content_type': request.content_type,
            'data': request.get_json(silent=True) or request.form.to_dict()
        })
    
    return jsonify({
        'method': 'GET',
        'args': request.args.to_dict()
    })

# Query parameters
@app.route('/search')
def search():
    """Handle query parameters."""
    query = request.args.get('q', '')
    category = request.args.get('category', 'all')
    return jsonify({
        'query': query,
        'category': category,
        'message': f'Searching for "{query}" in category "{category}"'
    })

# Multiple routes for same function
@app.route('/about')
@app.route('/about/<lang>')
def about(lang='en'):
    """Multiple routes with optional parameter."""
    messages = {
        'en': 'Welcome to our Flask demo!',
        'es': '¡Bienvenido a nuestra demostración de Flask!',
        'fr': 'Bienvenue dans notre démo Flask!'
    }
    return messages.get(lang, messages['en'])

# Error handling
@app.errorhandler(404)
def not_found(error):
    """Custom 404 error handler."""
    return jsonify({
        'error': 'Not Found',
        'message': 'The requested resource was not found'
    }), 404

if __name__ == '__main__':
    # Run the application in debug mode
    app.run(debug=True) 