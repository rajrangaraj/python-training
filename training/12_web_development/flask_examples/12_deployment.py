"""
Demonstration of Flask deployment configuration and production setup.
"""

from flask import Flask
from werkzeug.middleware.proxy_fix import ProxyFix
from werkzeug.contrib.fixers import LighttpdCGIRootFix
import os
import logging
from logging.handlers import RotatingFileHandler, SysLogHandler
import sentry_sdk
from sentry_sdk.integrations.flask import FlaskIntegration
from prometheus_flask_exporter import PrometheusMetrics
import newrelic.agent

# Initialize Sentry SDK for error tracking
sentry_sdk.init(
    dsn="your-sentry-dsn",
    integrations=[FlaskIntegration()],
    traces_sample_rate=1.0,
    environment=os.getenv('FLASK_ENV', 'production')
)

app = Flask(__name__)

# Production configuration
app.config.update(
    ENV='production',
    DEBUG=False,
    TESTING=False,
    SECRET_KEY=os.environ['SECRET_KEY'],
    SERVER_NAME=os.environ.get('SERVER_NAME'),
    APPLICATION_ROOT='/',
    PREFERRED_URL_SCHEME='https',
    
    # Database
    SQLALCHEMY_DATABASE_URI=os.environ['DATABASE_URL'],
    SQLALCHEMY_POOL_SIZE=10,
    SQLALCHEMY_MAX_OVERFLOW=20,
    SQLALCHEMY_POOL_TIMEOUT=30,
    
    # Cache
    CACHE_TYPE='redis',
    CACHE_REDIS_URL=os.environ['REDIS_URL'],
    CACHE_DEFAULT_TIMEOUT=300,
    
    # Session
    SESSION_TYPE='redis',
    SESSION_REDIS=os.environ['REDIS_URL'],
    PERMANENT_SESSION_LIFETIME=86400,  # 24 hours
    
    # Mail
    MAIL_SERVER=os.environ['MAIL_SERVER'],
    MAIL_PORT=int(os.environ['MAIL_PORT']),
    MAIL_USE_TLS=True,
    MAIL_USERNAME=os.environ['MAIL_USERNAME'],
    MAIL_PASSWORD=os.environ['MAIL_PASSWORD'],
    
    # File uploads
    MAX_CONTENT_LENGTH=16 * 1024 * 1024,  # 16MB
    UPLOAD_FOLDER='/var/www/uploads'
)

# Configure middleware for proxy servers
app.wsgi_app = ProxyFix(
    app.wsgi_app,
    x_for=1,
    x_proto=1,
    x_host=1,
    x_port=1,
    x_prefix=1
)

# Initialize metrics
metrics = PrometheusMetrics(app)
metrics.info('app_info', 'Application info', version='1.0.0')

# Configure logging
if not app.debug:
    # File logging
    file_handler = RotatingFileHandler(
        'logs/production.log',
        maxBytes=10485760,  # 10MB
        backupCount=10
    )
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s %(levelname)s: %(message)s '
        '[in %(pathname)s:%(lineno)d]'
    ))
    file_handler.setLevel(logging.INFO)
    app.logger.addHandler(file_handler)
    
    # Syslog
    syslog_handler = SysLogHandler()
    syslog_handler.setLevel(logging.WARNING)
    app.logger.addHandler(syslog_handler)
    
    app.logger.setLevel(logging.INFO)
    app.logger.info('Production startup')

# Health check endpoint
@app.route('/health')
@metrics.do_not_track()
def health_check():
    """Health check endpoint for load balancers."""
    return {
        'status': 'healthy',
        'version': '1.0.0',
        'environment': app.config['ENV']
    }

# Error tracking
@app.errorhandler(Exception)
def handle_error(error):
    """Track all errors in Sentry."""
    sentry_sdk.capture_exception(error)
    app.logger.error(f'Unhandled error: {error}')
    return 'Internal Server Error', 500

# Gunicorn configuration
bind = '0.0.0.0:8000'
workers = 4  # (2 x num_cores) + 1
worker_class = 'gevent'
worker_connections = 1000
timeout = 30
keepalive = 2

# Nginx configuration example
NGINX_CONF = """
server {
    listen 80;
    server_name example.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name example.com;

    ssl_certificate /etc/letsencrypt/live/example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/example.com/privkey.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;

    access_log /var/log/nginx/access.log;
    error_log /var/log/nginx/error.log;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    location /static {
        alias /var/www/static;
        expires 30d;
        add_header Cache-Control "public, no-transform";
    }

    location /uploads {
        internal;
        alias /var/www/uploads;
    }
}
"""

# Supervisor configuration example
SUPERVISOR_CONF = """
[program:flask_app]
directory=/var/www/app
command=/var/www/app/venv/bin/gunicorn -c gunicorn_config.py wsgi:app
user=www-data
autostart=true
autorestart=true
stopasgroup=true
killasgroup=true
stderr_logfile=/var/log/supervisor/flask_app.err.log
stdout_logfile=/var/log/supervisor/flask_app.out.log
"""

# Docker configuration
DOCKERFILE = """
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV FLASK_APP=app.py
ENV FLASK_ENV=production

EXPOSE 8000

CMD ["gunicorn", "-c", "gunicorn_config.py", "wsgi:app"]
"""

if __name__ == '__main__':
    app.run(ssl_context='adhoc') 