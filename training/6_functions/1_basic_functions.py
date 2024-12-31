"""
Introduction to basic function definitions and calls in Python.
"""

# Simple function with no parameters
def greet():
    """Simple greeting function."""
    print("Hello, World!")

# Function with parameters
def greet_person(name):
    """Greet a specific person."""
    print(f"Hello, {name}!")

# Function with default parameter
def greet_with_time(name, time_of_day="morning"):
    """Greet person with time of day."""
    print(f"Good {time_of_day}, {name}!")

# Function that performs calculation
def add_numbers(a, b):
    """Add two numbers and return result."""
    return a + b

# Function with type hints (Python 3.5+)
def multiply(x: float, y: float) -> float:
    """Multiply two numbers with type hints."""
    return x * y

# Function with multiple parameters
def describe_person(name, age, city):
    """Describe a person's basic information."""
    print(f"{name} is {age} years old and lives in {city}.")

# Example usage
if __name__ == "__main__":
    # Basic function calls
    print("Basic function calls:")
    greet()
    greet_person("Alice")
    greet_with_time("Bob", "evening")
    
    # Functions with return values
    print("\nFunction return values:")
    result = add_numbers(5, 3)
    print(f"5 + 3 = {result}")
    
    product = multiply(4.0, 2.5)
    print(f"4.0 * 2.5 = {product}")
    
    # Multiple parameters
    print("\nMultiple parameters:")
    describe_person("Charlie", 25, "New York") 