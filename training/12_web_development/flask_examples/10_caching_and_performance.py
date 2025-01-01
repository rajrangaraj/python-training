"""
Demonstration of Flask caching and performance optimization.
"""

from flask import Flask, jsonify, request
from flask_caching import Cache
from flask_sqlalchemy import SQLAlchemy
from werkzeug.middleware.proxy_fix import ProxyFix
from datetime import datetime
import functools
import time

app = Flask(__name__)

# Cache configuration
cache_config = {
    'CACHE_TYPE': 'redis',
    'CACHE_REDIS_URL': 'redis://localhost:6379/0',
    'CACHE_DEFAULT_TIMEOUT': 300  # 5 minutes
}
app.config.update(cache_config)

# Database configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///cache_demo.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize extensions
cache = Cache(app)
db = SQLAlchemy(app)

# Add ProxyFix middleware for proper header handling
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1)

# Models
class Product(db.Model):
    """Product model with caching."""
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    price = db.Column(db.Float, nullable=False)
    stock = db.Column(db.Integer, nullable=False)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow)

# Custom cache decorators
def timed_cache(timeout=300):
    """Custom cache decorator with timing."""
    def decorator(f):
        @functools.wraps(f)
        def decorated_function(*args, **kwargs):
            cache_key = f'{f.__name__}:{str(args)}:{str(kwargs)}'
            start_time = time.time()
            
            # Try to get from cache
            rv = cache.get(cache_key)
            if rv is not None:
                end_time = time.time()
                app.logger.info(f'Cache hit for {cache_key}. Time: {end_time - start_time:.4f}s')
                return rv
            
            # If not in cache, call function
            rv = f(*args, **kwargs)
            cache.set(cache_key, rv, timeout=timeout)
            
            end_time = time.time()
            app.logger.info(f'Cache miss for {cache_key}. Time: {end_time - start_time:.4f}s')
            return rv
        return decorated_function
    return decorator

def cache_invalidate(keys):
    """Decorator to invalidate cache keys."""
    def decorator(f):
        @functools.wraps(f)
        def decorated_function(*args, **kwargs):
            rv = f(*args, **kwargs)
            for key in keys:
                cache.delete(key)
            return rv
        return decorated_function
    return decorator

# Routes
@app.route('/api/products')
@timed_cache(timeout=60)
def get_products():
    """Get all products with caching."""
    products = Product.query.all()
    return jsonify([{
        'id': p.id,
        'name': p.name,
        'price': p.price,
        'stock': p.stock,
        'updated_at': p.updated_at.isoformat()
    } for p in products])

@app.route('/api/products/<int:product_id>')
@cache.memoize(60)
def get_product(product_id):
    """Get single product with caching."""
    product = Product.query.get_or_404(product_id)
    return jsonify({
        'id': product.id,
        'name': product.name,
        'price': product.price,
        'stock': product.stock,
        'updated_at': product.updated_at.isoformat()
    })

@app.route('/api/products', methods=['POST'])
@cache_invalidate(['get_products'])
def create_product():
    """Create new product and invalidate cache."""
    data = request.get_json()
    
    product = Product(
        name=data['name'],
        price=float(data['price']),
        stock=int(data['stock'])
    )
    
    db.session.add(product)
    db.session.commit()
    
    return jsonify({
        'id': product.id,
        'name': product.name,
        'price': product.price,
        'stock': product.stock,
        'updated_at': product.updated_at.isoformat()
    }), 201

@app.route('/api/products/<int:product_id>', methods=['PUT'])
@cache_invalidate(['get_products'])
def update_product(product_id):
    """Update product and invalidate cache."""
    product = Product.query.get_or_404(product_id)
    data = request.get_json()
    
    if 'name' in data:
        product.name = data['name']
    if 'price' in data:
        product.price = float(data['price'])
    if 'stock' in data:
        product.stock = int(data['stock'])
    
    product.updated_at = datetime.utcnow()
    db.session.commit()
    
    # Invalidate specific product cache
    cache.delete_memoized(get_product, product_id)
    
    return jsonify({
        'id': product.id,
        'name': product.name,
        'price': product.price,
        'stock': product.stock,
        'updated_at': product.updated_at.isoformat()
    })

# Performance monitoring
@app.before_request
def start_timer():
    """Start timing request."""
    request._start_time = time.time()

@app.after_request
def log_request(response):
    """Log request timing."""
    if hasattr(request, '_start_time'):
        total_time = time.time() - request._start_time
        app.logger.info(f'Request to {request.path} took {total_time:.4f}s')
    return response

# CLI commands
@app.cli.command('cache-clear')
def cache_clear():
    """Clear the cache."""
    cache.clear()
    print('Cache cleared.')

@app.cli.command('cache-stats')
def cache_stats():
    """Show cache statistics."""
    stats = cache.get_stats()
    print('Cache statistics:')
    for key, value in stats.items():
        print(f'{key}: {value}')

if __name__ == '__main__':
    app.run(debug=True) 