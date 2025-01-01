"""
Demonstration of basic unit testing in Python using unittest framework.
"""

import unittest
from typing import List, Optional

# Class to be tested
class Calculator:
    """A simple calculator class to demonstrate unit testing."""
    
    def add(self, a: float, b: float) -> float:
        """Add two numbers."""
        return a + b
    
    def subtract(self, a: float, b: float) -> float:
        """Subtract b from a."""
        return a - b
    
    def multiply(self, a: float, b: float) -> float:
        """Multiply two numbers."""
        return a * b
    
    def divide(self, a: float, b: float) -> Optional[float]:
        """Divide a by b. Returns None if b is zero."""
        try:
            return a / b
        except ZeroDivisionError:
            return None
    
    def average(self, numbers: List[float]) -> Optional[float]:
        """Calculate the average of a list of numbers."""
        if not numbers:
            return None
        return sum(numbers) / len(numbers)

# Test class
class TestCalculator(unittest.TestCase):
    """Test cases for Calculator class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.calc = Calculator()
    
    def test_add(self):
        """Test addition."""
        self.assertEqual(self.calc.add(3, 5), 8)
        self.assertEqual(self.calc.add(-1, 1), 0)
        self.assertEqual(self.calc.add(0, 0), 0)
        self.assertEqual(self.calc.add(3.5, 2.5), 6.0)
    
    def test_subtract(self):
        """Test subtraction."""
        self.assertEqual(self.calc.subtract(5, 3), 2)
        self.assertEqual(self.calc.subtract(1, 1), 0)
        self.assertEqual(self.calc.subtract(0, 5), -5)
        self.assertEqual(self.calc.subtract(3.5, 2.5), 1.0)
    
    def test_multiply(self):
        """Test multiplication."""
        self.assertEqual(self.calc.multiply(3, 5), 15)
        self.assertEqual(self.calc.multiply(-2, 3), -6)
        self.assertEqual(self.calc.multiply(0, 5), 0)
        self.assertEqual(self.calc.multiply(2.5, 2), 5.0)
    
    def test_divide(self):
        """Test division."""
        self.assertEqual(self.calc.divide(6, 2), 3)
        self.assertEqual(self.calc.divide(5, 2), 2.5)
        self.assertEqual(self.calc.divide(0, 5), 0)
        self.assertIsNone(self.calc.divide(5, 0))
    
    def test_average(self):
        """Test average calculation."""
        self.assertEqual(self.calc.average([1, 2, 3]), 2)
        self.assertEqual(self.calc.average([0, 0, 0]), 0)
        self.assertEqual(self.calc.average([1.5, 2.5]), 2.0)
        self.assertIsNone(self.calc.average([]))
    
    def test_floating_point(self):
        """Test floating point calculations."""
        self.assertAlmostEqual(self.calc.add(0.1, 0.2), 0.3, places=7)
        self.assertAlmostEqual(self.calc.multiply(0.1, 0.3), 0.03, places=7)
    
    def tearDown(self):
        """Clean up after each test method."""
        pass

# Additional test class demonstrating test organization
class TestCalculatorEdgeCases(unittest.TestCase):
    """Test edge cases for Calculator class."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures once for all test methods in the class."""
        cls.calc = Calculator()
    
    def test_large_numbers(self):
        """Test calculations with large numbers."""
        large_num = 10**15
        self.assertEqual(self.calc.add(large_num, large_num), 2 * large_num)
        self.assertEqual(self.calc.multiply(large_num, 0), 0)
    
    def test_type_errors(self):
        """Test that type errors are raised appropriately."""
        with self.assertRaises(TypeError):
            self.calc.add("1", 2)
        
        with self.assertRaises(TypeError):
            self.calc.multiply(None, 3)
        
        with self.assertRaises(TypeError):
            self.calc.average(["1", "2", "3"])
    
    @unittest.skip("Demonstrating test skipping")
    def test_skipped(self):
        """This test will be skipped."""
        self.fail("This test should be skipped")
    
    @unittest.expectedFailure
    def test_expected_failure(self):
        """This test is expected to fail."""
        self.assertEqual(0.1 + 0.2, 0.3)  # Will fail due to floating point precision

if __name__ == '__main__':
    unittest.main() 