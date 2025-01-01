"""
Demonstration of Flask application structure using blueprints.
"""

from flask import Flask
from config import Config
from extensions import db, migrate, login_manager
from blueprints.auth import auth_bp
from blueprints.main import main_bp
from blueprints.api import api_bp
from blueprints.admin import admin_bp

def create_app(config_class=Config):
    """Application factory function."""
    app = Flask(__name__)
    app.config.from_object(config_class)
    
    # Initialize extensions
    db.init_app(app)
    migrate.init_app(app, db)
    login_manager.init_app(app)
    
    # Register blueprints
    app.register_blueprint(auth_bp, url_prefix='/auth')
    app.register_blueprint(main_bp)
    app.register_blueprint(api_bp, url_prefix='/api/v1')
    app.register_blueprint(admin_bp, url_prefix='/admin')
    
    # Configure logging
    if not app.debug and not app.testing:
        configure_logging(app)
    
    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True) 