"""
Example of a custom module for basic calculations.
"""

def add(a, b):
    """Add two numbers."""
    return a + b

def subtract(a, b):
    """Subtract b from a."""
    return a - b

def multiply(a, b):
    """Multiply two numbers."""
    return a * b

def divide(a, b):
    """Divide a by b."""
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b

# Constants
PI = 3.14159
E = 2.71828

# Private function (by convention)
def _validate_numbers(*args):
    """Validate that all arguments are numbers."""
    return all(isinstance(arg, (int, float)) for arg in args)

# Example class
class Calculator:
    """A simple calculator class."""
    
    def __init__(self):
        self.history = []
    
    def calculate(self, operation, a, b):
        """Perform calculation and store in history."""
        if not _validate_numbers(a, b):
            raise ValueError("Invalid number type")
        
        if operation == 'add':
            result = add(a, b)
        elif operation == 'subtract':
            result = subtract(a, b)
        elif operation == 'multiply':
            result = multiply(a, b)
        elif operation == 'divide':
            result = divide(a, b)
        else:
            raise ValueError("Invalid operation")
        
        self.history.append((operation, a, b, result))
        return result
    
    def get_history(self):
        """Return calculation history."""
        return self.history

# Example usage if run directly
if __name__ == "__main__":
    # Test basic functions
    print(f"2 + 3 = {add(2, 3)}")
    print(f"5 - 2 = {subtract(5, 2)}")
    print(f"4 * 6 = {multiply(4, 6)}")
    print(f"8 / 2 = {divide(8, 2)}")
    
    # Test calculator class
    calc = Calculator()
    print("\nCalculator class test:")
    print(f"3 + 4 = {calc.calculate('add', 3, 4)}")
    print(f"History: {calc.get_history()}") 