"""
Demonstration of Flask templates and static file handling.
"""

from flask import Flask, render_template, url_for
from datetime import datetime

app = Flask(__name__)

# Sample data
PRODUCTS = [
    {'id': 1, 'name': 'Laptop', 'price': 999.99, 'in_stock': True},
    {'id': 2, 'name': 'Mouse', 'price': 29.99, 'in_stock': True},
    {'id': 3, 'name': 'Monitor', 'price': 299.99, 'in_stock': False},
    {'id': 4, 'name': 'Keyboard', 'price': 59.99, 'in_stock': True}
]

@app.route('/')
def home():
    """Home page using template."""
    return render_template(
        'home.html',
        title='Welcome',
        current_year=datetime.now().year
    )

@app.route('/products')
def products():
    """Product listing page."""
    return render_template(
        'products.html',
        title='Our Products',
        products=PRODUCTS
    )

@app.route('/product/<int:product_id>')
def product_detail(product_id):
    """Product detail page."""
    product = next(
        (p for p in PRODUCTS if p['id'] == product_id),
        None
    )
    if product:
        return render_template(
            'product_detail.html',
            title=product['name'],
            product=product
        )
    return render_template('404.html'), 404

@app.context_processor
def utility_processor():
    """Add utility functions to template context."""
    def format_price(price):
        """Format price with currency symbol."""
        return f'${price:.2f}'
    
    return dict(format_price=format_price)

@app.template_filter('status_badge')
def status_badge(in_stock):
    """Custom template filter for stock status."""
    return 'In Stock' if in_stock else 'Out of Stock'

if __name__ == '__main__':
    app.run(debug=True) 