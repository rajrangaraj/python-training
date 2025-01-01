"""
Main blueprint routes.
"""

from flask import Blueprint, render_template, current_app
from flask_login import login_required
from extensions import cache, limiter

main = Blueprint('main', __name__)

@main.route('/')
@cache.cached(timeout=300)  # Cache for 5 minutes
def index():
    """Home page."""
    return render_template('main/index.html')

@main.route('/dashboard')
@login_required
@limiter.limit("100 per minute")
def dashboard():
    """User dashboard."""
    return render_template('main/dashboard.html')

@main.route('/about')
@cache.cached(timeout=3600)  # Cache for 1 hour
def about():
    """About page."""
    return render_template('main/about.html') 