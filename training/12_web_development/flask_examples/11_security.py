"""
Demonstration of Flask security features and best practices.
"""

from flask import Flask, request, render_template, session, redirect, url_for
from flask_talisman import Talisman
from flask_seasurf import SeaSurf
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import secrets
from datetime import timedelta
import logging
from logging.handlers import RotatingFileHandler
import re
import bleach
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)

# Security configuration
app.config.update(
    SECRET_KEY=secrets.token_hex(32),
    SESSION_COOKIE_SECURE=True,
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE='Lax',
    PERMANENT_SESSION_LIFETIME=timedelta(days=7),
    MAX_CONTENT_LENGTH=16 * 1024 * 1024  # 16MB max file size
)

# Initialize security extensions
talisman = Talisman(
    app,
    force_https=True,
    strict_transport_security=True,
    session_cookie_secure=True,
    content_security_policy={
        'default-src': "'self'",
        'script-src': ["'self'", "'unsafe-inline'"],
        'style-src': ["'self'", "'unsafe-inline'"],
        'img-src': ["'self'", 'data:', 'https:'],
        'font-src': ["'self'", 'https:', 'data:'],
    }
)

csrf = SeaSurf(app)
limiter = Limiter(
    app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"]
)

# Security helpers
def sanitize_input(text):
    """Sanitize user input."""
    return bleach.clean(
        text,
        tags=['p', 'b', 'i', 'u', 'em', 'strong'],
        attributes={},
        strip=True
    )

def validate_password(password):
    """Validate password strength."""
    if len(password) < 12:
        return False, "Password must be at least 12 characters long"
    
    patterns = [
        (r'[A-Z]', "uppercase letter"),
        (r'[a-z]', "lowercase letter"),
        (r'[0-9]', "number"),
        (r'[^A-Za-z0-9]', "special character")
    ]
    
    missing = [msg for pattern, msg in patterns if not re.search(pattern, password)]
    
    if missing:
        return False, f"Password must contain at least one {', '.join(missing)}"
    
    return True, "Password is strong"

# Routes with security features
@app.route('/login', methods=['GET', 'POST'])
@limiter.limit("5 per minute")
def login():
    """Secure login route."""
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        
        # Implement constant-time comparison for passwords
        if not secrets.compare_digest(username, 'admin') or \
           not check_password_hash(stored_hash, password):
            app.logger.warning(f'Failed login attempt for user: {username}')
            return render_template('login.html', error="Invalid credentials"), 401
        
        session.permanent = True
        session['user_id'] = 1
        session['last_login'] = datetime.utcnow().isoformat()
        
        return redirect(url_for('dashboard'))
    
    return render_template('login.html')

@app.route('/api/comment', methods=['POST'])
@csrf.exempt  # Only for API endpoints that need CSRF exemption
@limiter.limit("10 per minute")
def add_comment():
    """Secure comment submission."""
    if not request.is_json:
        return {'error': 'Content-Type must be application/json'}, 400
    
    data = request.get_json()
    comment = data.get('comment', '').strip()
    
    if not comment:
        return {'error': 'Comment cannot be empty'}, 400
    
    # Sanitize input
    clean_comment = sanitize_input(comment)
    
    # Store comment securely
    # ... database operations ...
    
    return {'status': 'success', 'comment': clean_comment}, 201

@app.route('/upload', methods=['POST'])
def upload_file():
    """Secure file upload."""
    if 'file' not in request.files:
        return {'error': 'No file provided'}, 400
    
    file = request.files['file']
    
    if not file.filename:
        return {'error': 'No file selected'}, 400
    
    # Validate file type
    allowed_extensions = {'png', 'jpg', 'jpeg', 'gif', 'pdf'}
    if not ('.' in file.filename and 
            file.filename.rsplit('.', 1)[1].lower() in allowed_extensions):
        return {'error': 'Invalid file type'}, 400
    
    # Generate secure filename
    filename = secrets.token_hex(16) + '.' + file.filename.rsplit('.', 1)[1].lower()
    
    # Save file securely
    try:
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    except Exception as e:
        app.logger.error(f'File upload error: {e}')
        return {'error': 'Error saving file'}, 500
    
    return {'status': 'success', 'filename': filename}, 201

# Error handlers
@app.errorhandler(429)
def ratelimit_handler(e):
    """Handle rate limit exceeded."""
    return render_template('error.html',
        error="Rate limit exceeded. Please try again later."), 429

# Logging configuration
if not app.debug:
    file_handler = RotatingFileHandler(
        'logs/security.log',
        maxBytes=10240,
        backupCount=10
    )
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s %(levelname)s: %(message)s '
        '[in %(pathname)s:%(lineno)d]'
    ))
    file_handler.setLevel(logging.INFO)
    app.logger.addHandler(file_handler)
    app.logger.setLevel(logging.INFO)
    app.logger.info('Application startup')

# Security headers middleware
@app.after_request
def add_security_headers(response):
    """Add security headers to response."""
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'SAMEORIGIN'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
    response.headers['Permissions-Policy'] = 'geolocation=(), microphone=()'
    return response

if __name__ == '__main__':
    app.run(ssl_context='adhoc')  # Enable HTTPS in development 