"""
Demonstration of function return values and multiple returns in Python.
"""

# Single return value
def square(number):
    """Return the square of a number."""
    return number ** 2

# Multiple return values
def get_coordinates():
    """Return multiple values as a tuple."""
    x = 10
    y = 20
    z = 30
    return x, y, z  # Returns a tuple

# Returning different types
def process_number(num):
    """Return different types based on conditions."""
    if num < 0:
        return None
    elif num == 0:
        return False
    else:
        return str(num)

# Returning collections
def create_user(name, age):
    """Return a dictionary with user information."""
    return {
        "name": name,
        "age": age,
        "created_at": "2024-01-01"
    }

# Return with early exit
def find_index(items, target):
    """Return index of target item or -1 if not found."""
    for i, item in enumerate(items):
        if item == target:
            return i
    return -1

# Return multiple results as dictionary
def analyze_text(text):
    """Return multiple text analysis results."""
    return {
        "length": len(text),
        "words": len(text.split()),
        "uppercase": sum(1 for c in text if c.isupper()),
        "lowercase": sum(1 for c in text if c.islower()),
        "digits": sum(1 for c in text if c.isdigit())
    }

# Example usage
if __name__ == "__main__":
    # Single return value
    print("Single return value:")
    result = square(5)
    print(f"Square of 5: {result}")
    
    # Multiple return values
    print("\nMultiple return values:")
    x, y, z = get_coordinates()
    print(f"Coordinates: ({x}, {y}, {z})")
    
    # Different return types
    print("\nDifferent return types:")
    print(f"Process -1: {process_number(-1)}")
    print(f"Process 0: {process_number(0)}")
    print(f"Process 5: {process_number(5)}")
    
    # Returning collections
    print("\nReturning collections:")
    user = create_user("Alice", 30)
    print(f"User info: {user}")
    
    # Early return
    print("\nEarly return:")
    numbers = [1, 2, 3, 4, 5]
    index = find_index(numbers, 3)
    print(f"Index of 3: {index}")
    
    # Multiple results
    print("\nText analysis:")
    analysis = analyze_text("Hello World 123!")
    for key, value in analysis.items():
        print(f"{key}: {value}") 