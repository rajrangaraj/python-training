"""
Demonstration of Flask authentication, user sessions, and security.
"""

from flask import (
    Flask, render_template, request, redirect, url_for, 
    flash, session, g
)
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
import os
from datetime import datetime, timedelta

app = Flask(__name__)
app.secret_key = os.urandom(24)  # Better secret key generation
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=7)  # Session duration

db = SQLAlchemy(app)

# User model with additional security features
class User(db.Model):
    """User model with security features."""
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)
    is_active = db.Column(db.Boolean, default=True)
    is_admin = db.Column(db.Boolean, default=False)
    last_login = db.Column(db.DateTime)
    failed_login_attempts = db.Column(db.Integer, default=0)
    locked_until = db.Column(db.DateTime)
    
    def set_password(self, password):
        """Hash and set the password."""
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        """Check if password is correct."""
        return check_password_hash(self.password_hash, password)
    
    def is_locked(self):
        """Check if account is locked."""
        if self.locked_until and self.locked_until > datetime.utcnow():
            return True
        return False

# Decorators for authentication and authorization
def login_required(f):
    """Decorator to require login for views."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in to access this page.', 'error')
            return redirect(url_for('login', next=request.url))
        return f(*args, **kwargs)
    return decorated_function

def admin_required(f):
    """Decorator to require admin privileges."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not g.user or not g.user.is_admin:
            flash('Admin privileges required.', 'error')
            return redirect(url_for('index'))
        return f(*args, **kwargs)
    return decorated_function

# Request handlers
@app.before_request
def load_user():
    """Load user object before each request."""
    g.user = None
    if 'user_id' in session:
        g.user = User.query.get(session['user_id'])

# Authentication routes
@app.route('/login', methods=['GET', 'POST'])
def login():
    """Handle user login."""
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()
        
        if user and user.is_locked():
            flash('Account is locked. Please try again later.', 'error')
            return render_template('auth/login.html')
        
        if user and user.check_password(password):
            if not user.is_active:
                flash('Account is deactivated.', 'error')
                return render_template('auth/login.html')
            
            # Reset failed login attempts
            user.failed_login_attempts = 0
            user.last_login = datetime.utcnow()
            db.session.commit()
            
            session.permanent = True  # Use permanent session
            session['user_id'] = user.id
            
            next_page = request.args.get('next')
            if next_page:
                return redirect(next_page)
            return redirect(url_for('index'))
        else:
            if user:
                # Increment failed login attempts
                user.failed_login_attempts += 1
                if user.failed_login_attempts >= 3:
                    user.locked_until = datetime.utcnow() + timedelta(minutes=15)
                db.session.commit()
            
            flash('Invalid username or password.', 'error')
    
    return render_template('auth/login.html')

@app.route('/logout')
def logout():
    """Handle user logout."""
    session.pop('user_id', None)
    flash('You have been logged out.', 'success')
    return redirect(url_for('index'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    """Handle user registration."""
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']
        
        if password != confirm_password:
            flash('Passwords do not match.', 'error')
            return render_template('auth/register.html')
        
        if User.query.filter_by(username=username).first():
            flash('Username already exists.', 'error')
            return render_template('auth/register.html')
        
        if User.query.filter_by(email=email).first():
            flash('Email already registered.', 'error')
            return render_template('auth/register.html')
        
        user = User(username=username, email=email)
        user.set_password(password)
        
        try:
            db.session.add(user)
            db.session.commit()
            flash('Registration successful! Please log in.', 'success')
            return redirect(url_for('login'))
        except Exception as e:
            db.session.rollback()
            flash('Error during registration.', 'error')
    
    return render_template('auth/register.html')

# Protected routes
@app.route('/profile')
@login_required
def profile():
    """User profile page."""
    return render_template('auth/profile.html', user=g.user)

@app.route('/admin')
@login_required
@admin_required
def admin_panel():
    """Admin panel."""
    users = User.query.all()
    return render_template('auth/admin.html', users=users)

# Password reset functionality
@app.route('/reset-password', methods=['GET', 'POST'])
def reset_password():
    """Handle password reset request."""
    if request.method == 'POST':
        email = request.form['email']
        user = User.query.filter_by(email=email).first()
        
        if user:
            # In a real application, send password reset email
            flash('Password reset instructions sent to your email.', 'success')
        else:
            flash('Email not found.', 'error')
        
        return redirect(url_for('login'))
    
    return render_template('auth/reset_password.html')

if __name__ == '__main__':
    app.run(debug=True) 