"""
Demonstration of Behavior Driven Development (BDD) testing using pytest-bdd.
"""

from pytest_bdd import scenario, given, when, then, parsers
import pytest
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime

# Domain Models
@dataclass
class Product:
    """Product in the shopping system."""
    id: int
    name: str
    price: float
    stock: int

@dataclass
class CartItem:
    """Item in shopping cart."""
    product: Product
    quantity: int

class ShoppingCart:
    """Shopping cart implementation."""
    
    def __init__(self):
        self.items: List[CartItem] = []
    
    def add_item(self, product: Product, quantity: int = 1) -> None:
        """Add product to cart."""
        if quantity <= 0:
            raise ValueError("Quantity must be positive")
        if product.stock < quantity:
            raise ValueError("Not enough stock")
        
        # Update existing item or add new one
        for item in self.items:
            if item.product.id == product.id:
                new_quantity = item.quantity + quantity
                if product.stock < new_quantity:
                    raise ValueError("Not enough stock")
                item.quantity = new_quantity
                return
        
        self.items.append(CartItem(product, quantity))
    
    def remove_item(self, product_id: int) -> None:
        """Remove product from cart."""
        self.items = [item for item in self.items if item.product.id != product_id]
    
    def get_total(self) -> float:
        """Calculate total price."""
        return sum(item.product.price * item.quantity for item in self.items)
    
    def clear(self) -> None:
        """Clear all items from cart."""
        self.items.clear()

# Fixtures
@pytest.fixture
def product_catalog() -> Dict[str, Product]:
    """Provide sample products for testing."""
    return {
        "book": Product(1, "Python Testing Book", 29.99, 5),
        "laptop": Product(2, "Developer Laptop", 999.99, 2),
        "mouse": Product(3, "Wireless Mouse", 19.99, 10)
    }

@pytest.fixture
def cart() -> ShoppingCart:
    """Provide a fresh shopping cart."""
    return ShoppingCart()

# Step Definitions
@given("an empty shopping cart")
def empty_cart(cart: ShoppingCart):
    """Ensure cart is empty."""
    cart.clear()
    assert len(cart.items) == 0

@given(parsers.parse("a product catalog containing a {product_name}"))
def check_product_exists(product_catalog: Dict[str, Product], product_name: str):
    """Check that product exists in catalog."""
    assert product_name in product_catalog

@when(parsers.parse("I add {quantity:d} {product_name} to the cart"))
def add_product_to_cart(
    cart: ShoppingCart,
    product_catalog: Dict[str, Product],
    quantity: int,
    product_name: str
):
    """Add product to cart."""
    product = product_catalog[product_name]
    cart.add_item(product, quantity)

@then(parsers.parse("the cart should contain {quantity:d} items"))
def check_cart_quantity(cart: ShoppingCart, quantity: int):
    """Check total quantity in cart."""
    total_quantity = sum(item.quantity for item in cart.items)
    assert total_quantity == quantity

@then(parsers.parse("the total price should be ${expected_total:f}"))
def check_cart_total(cart: ShoppingCart, expected_total: float):
    """Check cart total price."""
    assert cart.get_total() == pytest.approx(expected_total)

# Scenarios
@scenario('features/shopping_cart.feature', 'Add single item to cart')
def test_add_single_item():
    """Test adding single item to cart."""
    pass

@scenario('features/shopping_cart.feature', 'Add multiple items to cart')
def test_add_multiple_items():
    """Test adding multiple items to cart."""
    pass

@scenario('features/shopping_cart.feature', 'Remove item from cart')
def test_remove_item():
    """Test removing item from cart."""
    pass

# Additional step definitions for error cases
@when(parsers.parse("I try to add {quantity:d} {product_name} to the cart"))
def try_add_product_to_cart(
    cart: ShoppingCart,
    product_catalog: Dict[str, Product],
    quantity: int,
    product_name: str,
    request
):
    """Try to add product to cart, capturing any errors."""
    try:
        product = product_catalog[product_name]
        cart.add_item(product, quantity)
        request.session['error'] = None
    except ValueError as e:
        request.session['error'] = str(e)

@then(parsers.parse('I should get a "{error_message}" error'))
def check_error_message(error_message: str, request):
    """Check error message."""
    assert request.session['error'] == error_message

# Error scenarios
@scenario('features/shopping_cart.feature', 'Add item with insufficient stock')
def test_add_insufficient_stock():
    """Test adding item with insufficient stock."""
    pass

@scenario('features/shopping_cart.feature', 'Add item with invalid quantity')
def test_add_invalid_quantity():
    """Test adding item with invalid quantity."""
    pass

if __name__ == '__main__':
    pytest.main([__file__]) 