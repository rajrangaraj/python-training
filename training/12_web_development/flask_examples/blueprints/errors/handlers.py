"""
Error handlers for the application.
"""

from flask import Blueprint, render_template
from werkzeug.exceptions import HTTPException

errors = Blueprint('errors', __name__)

@errors.app_errorhandler(404)
def not_found_error(error):
    """Handle 404 errors."""
    return render_template('errors/404.html'), 404

@errors.app_errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    return render_template('errors/500.html'), 500

@errors.app_errorhandler(HTTPException)
def handle_exception(error):
    """Handle all HTTP exceptions."""
    return render_template('errors/error.html', error=error), error.code 