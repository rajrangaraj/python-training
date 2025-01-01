"""
Demonstration of pytest features including parameterization, fixtures, and markers.
"""

import pytest
from typing import List, Dict, Any
from dataclasses import dataclass
from datetime import datetime, timedelta

# Classes to be tested
@dataclass
class Product:
    """Product class for testing."""
    id: int
    name: str
    price: float
    category: str

class ShoppingCart:
    """Shopping cart implementation for testing."""
    
    def __init__(self):
        self.items: List[Product] = []
    
    def add_item(self, product: Product, quantity: int = 1) -> None:
        """Add a product to the cart."""
        if quantity <= 0:
            raise ValueError("Quantity must be positive")
        for _ in range(quantity):
            self.items.append(product)
    
    def remove_item(self, product_id: int) -> None:
        """Remove a product from the cart."""
        self.items = [item for item in self.items if item.id != product_id]
    
    def get_total(self) -> float:
        """Calculate total price of items in cart."""
        return sum(item.price for item in self.items)
    
    def get_item_count(self) -> int:
        """Get total number of items in cart."""
        return len(self.items)

# Fixtures
@pytest.fixture
def sample_products() -> List[Product]:
    """Fixture providing sample products for testing."""
    return [
        Product(1, "Laptop", 999.99, "Electronics"),
        Product(2, "Mouse", 29.99, "Electronics"),
        Product(3, "Coffee", 9.99, "Beverages"),
        Product(4, "Book", 19.99, "Books")
    ]

@pytest.fixture
def cart() -> ShoppingCart:
    """Fixture providing a fresh shopping cart for each test."""
    return ShoppingCart()

@pytest.fixture
def populated_cart(cart: ShoppingCart, sample_products: List[Product]) -> ShoppingCart:
    """Fixture providing a cart with some items."""
    cart.add_item(sample_products[0])  # Laptop
    cart.add_item(sample_products[1], 2)  # 2 Mice
    return cart

# Test classes
class TestShoppingCart:
    """Test cases for ShoppingCart using pytest features."""
    
    @pytest.mark.parametrize("product_index, quantity, expected_count", [
        (0, 1, 1),  # Add 1 laptop
        (1, 3, 3),  # Add 3 mice
        (2, 5, 5),  # Add 5 coffees
    ])
    def test_add_item_parameterized(
        self, 
        cart: ShoppingCart,
        sample_products: List[Product],
        product_index: int,
        quantity: int,
        expected_count: int
    ):
        """Test adding items with different quantities."""
        cart.add_item(sample_products[product_index], quantity)
        assert cart.get_item_count() == expected_count
    
    @pytest.mark.parametrize("quantity", [-1, 0])
    def test_add_item_invalid_quantity(
        self,
        cart: ShoppingCart,
        sample_products: List[Product],
        quantity: int
    ):
        """Test adding items with invalid quantities."""
        with pytest.raises(ValueError):
            cart.add_item(sample_products[0], quantity)
    
    def test_remove_item(self, populated_cart: ShoppingCart):
        """Test removing items from cart."""
        initial_count = populated_cart.get_item_count()
        populated_cart.remove_item(1)  # Remove laptop
        assert populated_cart.get_item_count() == initial_count - 1
    
    @pytest.mark.slow
    def test_large_cart_performance(self, cart: ShoppingCart, sample_products: List[Product]):
        """Test cart performance with many items."""
        for _ in range(1000):
            cart.add_item(sample_products[0])
        assert cart.get_item_count() == 1000
    
    @pytest.mark.skip(reason="Demonstrating skip marker")
    def test_skipped(self):
        """This test will be skipped."""
        assert False
    
    @pytest.mark.xfail(reason="Demonstrating expected failure")
    def test_expected_failure(self):
        """This test is expected to fail."""
        assert 0.1 + 0.2 == 0.3

class TestProductCategories:
    """Test cases demonstrating more pytest features."""
    
    @pytest.fixture(autouse=True)
    def setup_categories(self) -> Dict[str, List[Product]]:
        """Fixture that runs automatically for each test in this class."""
        self.categories = {
            "Electronics": [],
            "Beverages": [],
            "Books": []
        }
    
    def test_categorize_products(self, sample_products: List[Product]):
        """Test product categorization."""
        for product in sample_products:
            self.categories[product.category].append(product)
        
        assert len(self.categories["Electronics"]) == 2
        assert len(self.categories["Beverages"]) == 1
        assert len(self.categories["Books"]) == 1
    
    @pytest.mark.parametrize("category,expected_count", [
        ("Electronics", 2),
        ("Beverages", 1),
        ("Books", 1)
    ])
    def test_category_counts(
        self,
        sample_products: List[Product],
        category: str,
        expected_count: int
    ):
        """Test product counts per category."""
        for product in sample_products:
            self.categories[product.category].append(product)
        assert len(self.categories[category]) == expected_count

if __name__ == '__main__':
    pytest.main([__file__]) 